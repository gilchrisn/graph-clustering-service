
package scar

import (
    "encoding/json"
    "os"
    "time"
)

type MoveEvent struct {
    MoveNumber  int     `json:"move"`
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
}

func NewMoveTracker(filename, algorithm string) *MoveTracker {
    file, err := os.Create(filename)
    if err != nil {
        return nil
    }
    
    return &MoveTracker{
        file:      file,
        encoder:   json.NewEncoder(file),
        algorithm: algorithm,
    }
}

func (mt *MoveTracker) LogMove(moveNum, node, fromComm, toComm int, gain, modularity float64) {
    if mt == nil {
        return
    }
    
    event := MoveEvent{
        MoveNumber: moveNum,
        Algorithm:  mt.algorithm,
        Node:       node,
        FromComm:   fromComm,
        ToComm:     toComm,
        Gain:       gain,
        Modularity: modularity,
        Timestamp:  time.Now().Unix(),
    }
    
    mt.encoder.Encode(event)
}

func (mt *MoveTracker) Close() {
    if mt != nil && mt.file != nil {
        mt.file.Close()
    }
}
