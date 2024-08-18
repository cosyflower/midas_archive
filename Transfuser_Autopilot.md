
> Transfuser expert 생성 과정에 대해서 

- hd_map 데이터 받아들여서 관련된 경로 플래너 그리고 차량 데이터 초기화 
- world_map
- trajectory (전체 경로에서 위치 정보를 추출해서 밀집된 경로)
- dense_route (추출된 경로에서 더 세밀하게 보간한 밀집된 경로 - 정밀한 주행 경로를 제공한다)
- waypoint_planner (경로 플래너 초기화 - 보간된 경로를 사용해서 경로를 설정)
- waypoint_planner_extrapolation (예측 경로 플래너 초기화 - 경로 설정을 수행해서 경로 예측에 활용)
- command_planner : 더 먼 거리의 노드를 다루기 위한 경로 플래너, 전체 경로 계획에 사용된다

==sensors(self) : 기초적인 센서를 제외한 나머지, 특정된 센서들을 추가해서 반환 (dictionary)

==tick(self, input_data) < - 여기에서 말하는 input_data가 어떤 데이터인지를 파악할 필요가 있음
- gps(현 위치)
- speed(속도)
- compass (방위각)

==run_step(self, input_data, timestamp) <- 이 부분 핵심== (나중에 한번 더 파악하기)




==update_gps_buffer(self, control, theta, speed)

- theta - compass 정보를 그대로 반영한다
- control - steer, throttle, brake 정보를 가지고 있음
- GPS 위치를 업데이트 한다 
	- self.ego_model_gps.forward(변환된 좌표, yaw, speed, action)
	- 좌표를 다시 변환하고
	- gps_buffer에 반영한다 (다음 위치를 출력한다)


get_future_states(self) - self.future_states를 반환한다 

==def _get_control(self, input_data, steer=None, throttle=None,
vehicle_hazard=None, light_hazard=None, walker_hazard=None, stop_sign_hazard=None):

- hazard - true of false로 진행
- 모든 위험이 없다면 get_brake()로 현재 브레이크 값을 구한다
- 하나라도 존재한다면 브레이크를 활성화 한다