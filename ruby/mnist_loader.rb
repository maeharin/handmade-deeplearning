require 'zlib'

#
# mnistデータの構造: http://yann.lecun.com/exdb/mnist/
#
# usage:
#   x_train, t_train, x_test, t_test = load_mnist()
#

def load_mnist
  puts 'loading mnist...'

  x_train = load_images('../data/train-images-idx3-ubyte.gz')
  x_test = load_images('../data/t10k-images-idx3-ubyte.gz')
  t_train = load_labels('../data/train-labels-idx1-ubyte.gz')
  t_test = load_labels('../data/t10k-labels-idx1-ubyte.gz')

  puts 'loading mnist done'

  [x_train, t_train, x_test, t_test]
end

# 訓練用画像
# TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
# [offset] [type]          [value]          [description] 
# 0000     32 bit integer  0x00000803(2051) magic number 
# 0004     32 bit integer  60000            number of images 
# 0008     32 bit integer  28               number of rows 
# 0012     32 bit integer  28               number of columns 
# 0016     unsigned byte   ??               pixel 
# 0017     unsigned byte   ??               pixel 
# ........ 
# xxxx     unsigned byte   ??               pixel
# Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

# テスト用画像
# TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  10000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel
# Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
def load_images(file_name)
  Zlib::GzipReader.open(file_name) do |f|
    # メタデータ分をunpack
    # big endian unsigned 32bit(4byte) * 4
    # packテンプレートの文字列について: https://docs.ruby-lang.org/ja/2.2.0/doc/pack_template.html
    _, n_images, n_rows, n_cols = f.read(4 * 4).unpack('N4')

    # 実際の画像データを読み込みunpack
    n_images.times.map do
      # n_rows * n_cols分が画像1つ分
      binary = f.read(n_rows * n_cols)
      # C*: 8bit符号なし整数としてunpack
      binary.unpack('C*')
    end
  end
end

# TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
# [offset] [type]          [value]          [description] 
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
# 0004     32 bit integer  60000            number of items 
# 0008     unsigned byte   ??               label 
# 0009     unsigned byte   ??               label 
# ........ 
# xxxx     unsigned byte   ??               label
# The labels values are 0 to 9.
#
# TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  10000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label
# The labels values are 0 to 9.
def load_labels(file_name)
  Zlib::GzipReader.open(file_name) do |f|
    _, n_items = f.read(4 * 2).unpack('N2')
    n_items.times.map do
      binary = f.read(1)
      binary.unpack('C*')
    end
  end
end
