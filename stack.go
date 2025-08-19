package main

import "fmt"

type StackInfo struct {
	T   *ThreadInfo
	SID int32
}

func (s *StackInfo) QueryTraces() []string {
	return []string{fmt.Sprintf("stack trace for SID=%d", s.SID)}
}

func NewStackInfoFromEvent(ev Event, t *ThreadInfo) *StackInfo {
	return &StackInfo{
		T:   t,
		SID: ev.StackID,
	}
}
