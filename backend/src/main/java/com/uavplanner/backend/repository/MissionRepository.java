package com.uavplanner.backend.repository;

import com.uavplanner.backend.model.Mission;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MissionRepository extends JpaRepository<Mission,Long> {
}
