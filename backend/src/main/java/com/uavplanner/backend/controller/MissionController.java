package com.uavplanner.backend.controller;

import com.uavplanner.backend.model.Mission;
import com.uavplanner.backend.repository.MissionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/missions")
@CrossOrigin(origins = "http://localhost:3000")
public class MissionController {

    @Autowired
    private MissionRepository missionRepo;

    @Autowired
    private RestTemplate restTemplate;

    private static final String FLASK_URL = "https://uav-ml-service.onrender.com/predict-path";

    @PostMapping
    public ResponseEntity<?>createMission(@RequestBody Mission mission){
        try {
            if (mission.getOrigin() == null || mission.getDestination() == null) {
                return ResponseEntity.badRequest().body("Origin and Destination are required.");
            }
            Map<String, String> request = Map.of(
                    "origin", mission.getOrigin(),
                    "destination", mission.getDestination(),
                    "terrainType", mission.getTerrainType()
            );

            System.out.println("Calling Flask API with: " + request);
            Map<String, Object> resp = restTemplate.postForObject(FLASK_URL, request, Map.class);
            System.out.println("Flask Response: " + resp);

            if (resp == null || !resp.containsKey("predictedPath") || !resp.containsKey("shortestPath")) {
                return ResponseEntity.internalServerError().body("Invalid response from Flask");
            }
            mission.setPredictedPath(resp.get("predictedPath").toString());
            mission.setShortestPath(resp.get("shortestPath").toString());

            Mission saved = missionRepo.save(mission);

            Map<String,Object> responsePayload = Map.of(
                    "predictedPath", saved.getPredictedPath(),
                    "shortestPath", saved.getShortestPath(),
                    "missionName", saved.getMissionName()
            );
            return ResponseEntity.ok(responsePayload);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.internalServerError().body("Internal Server Error: " + e.getMessage());
        }
    }

    @GetMapping
    public List<Mission>getAllMissions(){
        return missionRepo.findAll();
    }
}
