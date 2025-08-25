package main

import "fmt"

type StackInfo struct {
	T   *ThreadInfo
	SID StackID
}

func (s *StackInfo) QueryTraces() []string {
	return []string{fmt.Sprintf("stack trace for SID=%d", s.SID)}
}
