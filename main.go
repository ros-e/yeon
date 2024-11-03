package main

import (
	"fmt"
	"narcissist/server"
)

func main() {
	fmt.Println("Welcome to Narcissist")
	fmt.Printf("Model: %s, Version: %s\n", Model, Version)
	server := server.CreateServer(":13726")
	server.Run()
}
