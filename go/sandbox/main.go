package main

import "fmt"

func main() {
	x := 0.2 * 1.0
	fmt.Printf("x: %v\n", x) // 0.2

	y := 0.8 * 0.5
	fmt.Printf("y: %v\n", y) // 0.4

	z := x + y
	fmt.Printf("z: %v\n", z) // 0.6000000000000001
	// なぜーーーーー

	if z != 0.6 {
		panic("not same!")
	} else {
		fmt.Println("success!")
	}
}
