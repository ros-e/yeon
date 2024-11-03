package cmd

import (
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use: "Narcissist",
	// Run: func(cmd *cobra.Command, args []string) {},
}

func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

func init() {
	rootCmd.PersistentFlags().BoolP("version", "v", false, "Show version information")
	rootCmd.CompletionOptions.DisableDefaultCmd = true
}
