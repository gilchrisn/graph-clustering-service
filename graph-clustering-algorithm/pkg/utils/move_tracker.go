package utils

import (
    "encoding/json"
    "fmt"
    "os"
    "time"
)

type MoveEvent struct {
    MoveNumber  int     `json:"move"`
    Level       int     `json:"level"`
    Algorithm   string  `json:"algorithm"`
    Node        int     `json:"node"`
    FromComm    int     `json:"from_comm"`
    ToComm      int     `json:"to_comm"`
    Gain        float64 `json:"gain"`
    Modularity  float64 `json:"modularity"`
    Timestamp   int64   `json:"timestamp"`
}

type MoveTracker struct {
    file      *os.File
    encoder   *json.Encoder
    algorithm string

    currentLevel   int    // Auto-detected level
    lastMoveNumber int    // Track for level detection
}


func NewMoveTracker(filename, algorithm string) *MoveTracker {
    file, err := os.Create(filename)
    if err != nil {
        fmt.Printf("❌ Failed to create file %s: %v\n", filename, err)
        return nil
    }
    
    return &MoveTracker{
        file:      file,
        encoder:   json.NewEncoder(file),
        algorithm: algorithm,

        currentLevel:   0,    // Start at level 0
        lastMoveNumber: 0,    // Track for drop detection
    }
}

func (mt *MoveTracker) LogMove(moveNum, node, fromComm, toComm int, gain, modularity float64) {
    if mt == nil {
        fmt.Printf("❌ MoveTracker is nil!\n")
        return
    }
    
    if moveNum < mt.lastMoveNumber {
        mt.currentLevel++
    }
    mt.lastMoveNumber = moveNum
    
    event := MoveEvent{
        MoveNumber: moveNum,
        Level:      mt.currentLevel,
        Algorithm:  mt.algorithm,
        Node:       node,
        FromComm:   fromComm,
        ToComm:     toComm,
        Gain:       gain,
        Modularity: modularity,
        Timestamp:  time.Now().Unix(),
    }
    
    // Check for encoding errors
    err := mt.encoder.Encode(event)
    if err != nil {
        fmt.Printf("❌ Failed to encode move: %v\n", err)
        return
    }
    
    // FORCE FLUSH TO DISK 
    err = mt.file.Sync()
    if err != nil {
        fmt.Printf("❌ Failed to sync file: %v\n", err)
    }
}

func (mt *MoveTracker) Close() {
    if mt != nil && mt.file != nil {
        fmt.Printf("🔒 Closing MoveTracker file\n")
        
        // Ensure final flush before closing
        mt.file.Sync()
        mt.file.Close()
        
        fmt.Printf("✅ MoveTracker file closed\n")
    }
}

 