"""Extract final scalar metrics from TensorBoardX event files."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from tensorboardX.proto import event_pb2


DEFAULT_TAGS = (
    "eval/episode_reward",
    "eval/episode_reward/success",
)


def read_events(path: Path):
  with path.open("rb") as stream:
    while length_bytes := stream.read(8):
      if len(length_bytes) != 8:
        raise ValueError(f"Truncated record length in {path}")
      length = struct.unpack("<Q", length_bytes)[0]
      if len(stream.read(4)) != 4:
        raise ValueError(f"Truncated length CRC in {path}")
      payload = stream.read(length)
      if len(payload) != length:
        raise ValueError(f"Truncated event payload in {path}")
      if len(stream.read(4)) != 4:
        raise ValueError(f"Truncated data CRC in {path}")
      yield event_pb2.Event.FromString(payload)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--logdir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--tag", action="append", dest="tags")
  args = parser.parse_args()
  tags = tuple(args.tags or DEFAULT_TAGS)

  event_files = sorted(args.logdir.rglob("events.out.tfevents.*"))
  if not event_files:
    raise FileNotFoundError(f"No TensorBoard event files under {args.logdir}")

  series: dict[str, list[dict[str, float | int]]] = {tag: [] for tag in tags}
  for event_file in event_files:
    for event in read_events(event_file):
      if not event.HasField("summary"):
        continue
      for value in event.summary.value:
        if value.tag in series and value.HasField("simple_value"):
          series[value.tag].append({
              "step": int(event.step),
              "value": float(value.simple_value),
              "wall_time": float(event.wall_time),
          })

  missing = [tag for tag, values in series.items() if not values]
  if missing:
    raise KeyError(f"Missing TensorBoard scalar tags: {missing}")

  for values in series.values():
    values.sort(key=lambda value: (value["step"], value["wall_time"]))

  final = {tag: values[-1] for tag, values in series.items()}
  report = {
      "event_files": [str(path) for path in event_files],
      "final": final,
      "series": series,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
