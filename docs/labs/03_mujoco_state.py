"""Inspect model/data/control boundaries in a tiny MuJoCo model."""

from __future__ import annotations

import numpy as np

import mujoco


XML = """
<mujoco model="slide_lab">
  <option timestep="0.005"/>
  <worldbody>
    <body name="block" pos="0 0 0.1">
      <joint name="slide" type="slide" axis="1 0 0" range="-1 1"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <site name="marker" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <position name="slide_position" joint="slide" kp="20" ctrlrange="-1 1"/>
  </actuator>
  <sensor>
    <jointpos name="slide_position_sensor" joint="slide"/>
  </sensor>
</mujoco>
"""


def main() -> None:
  model = mujoco.MjModel.from_xml_string(XML)
  data = mujoco.MjData(model)
  data.ctrl[0] = 0.25
  for _ in range(20):
    mujoco.mj_step(model, data)

  assert model.nq == model.nv == model.nu == 1
  assert np.isfinite(data.qpos).all()
  assert data.qpos[0] > 0.0
  sensor_id = model.sensor("slide_position_sensor").id
  sensor_address = model.sensor_adr[sensor_id]

  print(f"nq={model.nq} nv={model.nv} nu={model.nu}")
  print(f"qpos={data.qpos[0]:.6f} qvel={data.qvel[0]:.6f}")
  print(f"ctrl={data.ctrl[0]:.6f}")
  print(f"sensor_address={sensor_address} sensor={data.sensordata[sensor_address]:.6f}")


if __name__ == "__main__":
  main()
