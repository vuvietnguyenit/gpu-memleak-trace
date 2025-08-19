package main

import (
	"fmt"
	"os"
	"os/user"
	"syscall"
)

type ProcessInfo struct {
	PID  uint32
	Comm [16]byte
	UID  uint32
}

func NewProcessInfoFromEvent(ev Event) *ProcessInfo {
	return &ProcessInfo{
		PID:  ev.Pid,
		UID:  ev.Uid,
		Comm: ev.Comm, // remove nulls
	}
}

// Get full command by PID
func (p *ProcessInfo) FullCommand() (string, error) {
	cmdPath := fmt.Sprintf("/proc/%d/cmdline", p.PID)
	data, err := os.ReadFile(cmdPath)
	if err != nil {
		return "", err
	}
	// cmdline is null-separated
	return string(data), nil
}

// Get username of UID
func (p *ProcessInfo) Username() (string, error) {
	usr, err := user.LookupId(fmt.Sprint(p.UID))
	if err != nil {
		return "", err
	}
	return usr.Username, nil
}

// Check if PID exists
func (p *ProcessInfo) IsExists() bool {
	err := syscall.Kill(int(p.PID), 0)
	return err == nil
}
