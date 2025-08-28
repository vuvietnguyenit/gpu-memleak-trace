package main

import "sync"

type WG struct {
	sync.WaitGroup
}

func (wg *WG) Go(f func()) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		f()
	}()
}
