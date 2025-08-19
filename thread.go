package main

type ThreadInfo struct {
	P   *ProcessInfo
	TID uint32
}

func NewThreadInfoFromEvent(ev Event, p *ProcessInfo) *ThreadInfo {
	return &ThreadInfo{
		P:   p,
		TID: ev.Tid,
	}
}
