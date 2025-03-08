# Swarming-UAV

[![Latest Release](https://img.shields.io/github/v/release/zacharyzhu-ai/Swarming-UAV)](https://github.com/zacharyzhu-ai/Swarming-UAV/releases)

# Sentinel: Swarming UAVs with Computer Vision for Firefighters

## In Honor of Trevor Brown

This endeavor is dedicated to Trevor Brown, fellow volunteer firefighter.

On February 16, 2024: gas leak from a 500-pound propane tank ignited catastrophically, claiming his life with ten injuries from Sterling Fire, our neighboring station as they arrived on scene. As Maydays rang out, we responded https://www.loudoun.gov/CivicAlerts.aspx?AID=8957

## Project Sentinel: save those who risk their lives for others.

With nearly 100 fatalities and 70,000 injuries of first responders annually, this solution aims to revolutionize emergency response. Our goal is simple yet vital: ensure no first-responder faces preventable risks as they race to save others.

No anymore!

## Real-World Application

Two minutes before arrival, (my fire) Lt. Turner launches Sentinel's UAV swarm. Within one minute, the system:
- Detects intense rooftop flames
- Identifies explosive temperatures near ground floor stovetop
- Enables strategic positioning of:
  - Ladder truck to rooftop
  - Engine to southeast
  - Ambulance to east approach
- Ingress and egress recommendation

## Current Status
- Phase 1 - basics: Boids flocking algorithm implemented, and remotely run at Robotarium
- Take to flight: form-factor, platform, advanced algorithms

## Technical Overview

- **Language**: Python,
- **Computer Vision**: OpenCV
- **Drone Framework**: PX4/ArduPilot/Crazyflie
- **Simulation**: Gazebo
- **Testing Platform**: Robotarium and then?

## Roadmap

### Phase 1: basics, e.g. grounded robots/unicycle 
- Implement flocking algorithms with leader-follower per Boids https://en.wikipedia.org/wiki/Boids
- Develop fire detection simulation environment
- Create flock navigation towards detected hazards
- Deployed run remotely on Robotarium at Georgia Tech https://www.robotarium.gatech.edu/

### Phase 2: take to flight 
- Platform: PX4, ArduPilot or Crazyflie
- Form factor: how much miniaturized size vs. standard drones?
- Advanced algorithms:
  - Dynamic leader assignment
  - Multi-sensor integration (thermal, gas/CO2, hazmat)

### Phase 3: real-life scenarios
- Explosion detection and avoidance
- Situational and risk assessment
- Ingress and egress recommendation
