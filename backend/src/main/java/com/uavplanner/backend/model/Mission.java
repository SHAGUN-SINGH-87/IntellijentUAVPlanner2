package com.uavplanner.backend.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Mission {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String missionName;
    private String origin;
    private String destination;
    private String terrainType;
    @Lob
    private String predictedPath;
    @Lob
    private String shortestPath;
}
