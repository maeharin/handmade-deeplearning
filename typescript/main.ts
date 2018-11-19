const X = [
  [1,1],
  [0,0],
  [0,1],
  [1,0],
];
const T = [
  1,
  0,
  0,
  0
];

// 重み
// 2 x 3
let w1 = new Array(3)
for (let i = 0; i < 3; i++) {
  w1[i] = new Array(2)
  for (let j = 0; j < 2; j++) {
    w1[i][j] = 1
  }
}

// 3 x 1
let w2 = new Array()

// 
console.log(w1)