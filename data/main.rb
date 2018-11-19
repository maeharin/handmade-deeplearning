require 'zlib'

n_rows = n_cols = nil
images = []
labels = []

Zlib::GzipReader.open('t10k-images-idx3-ubyte.gz') do |f|
  magic, n_images = f.read(8).unpack('N2')
  raise 'This is not MNIST image file' if magic != 2051
  n_rows, n_cols = f.read(8).unpack('N2')
  n_images.times do
    images << f.read(n_rows * n_cols)
  end
end

p images[0].unpack('C*')
