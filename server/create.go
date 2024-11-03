package server

import (
	"fmt"
	"net/http"
)

type Server struct {
	addr string
}

func CreateServer(addr string) *Server {
	return &Server{
		addr: addr,
	}
}

func (s *Server) Run() {
router := http.NewServeMux()

server := http.Server{
	Addr: s.addr,
	Handler: router,
}
fmt.Printf("Server Started on %s", s.addr)
server.ListenAndServe()
}