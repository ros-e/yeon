package cmd

import (
	"Narcissist/server"
	"fmt"

	"github.com/spf13/cobra"
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Serves the API",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Server running. Press Ctrl+C to stop")
		server := server.CreateServer(":13726")
		server.Run()
	},
}

func init() {
	rootCmd.AddCommand(serveCmd)
}
