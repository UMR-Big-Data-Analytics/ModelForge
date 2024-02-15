import json
from datetime import datetime, time
from enum import Enum, unique
from typing import Dict, Union

import pandas as pd


@unique
class ScheduleEntryMode(Enum):

    REDUCED = "reduced"

    OFF = "off"

    ON = "on"

    NORMAL = "normal"

    FIXED = "fixed"

    TOP = "top"

    COMFORT = "comfort"

    TEMP_2 = "temp-2"

    STANDBY = "standby"

    LEVEL_ONE = "levelOne"

    LEVEL_TWO = "levelTwo"

    LEVEL_THREE = "levelThree"

    LEVEL_FOUR = "levelFour"


class Schedule:

    weekdays = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    mode_to_value = {
        # off / reduced / fixed / standby
        ScheduleEntryMode.REDUCED.value: 0,
        ScheduleEntryMode.OFF.value: 0,
        ScheduleEntryMode.FIXED.value: 0,
        ScheduleEntryMode.STANDBY.value: 0,
        # on / normal / top / level 1
        ScheduleEntryMode.ON.value: 1,
        ScheduleEntryMode.NORMAL.value: 1,
        ScheduleEntryMode.TOP.value: 1,
        ScheduleEntryMode.LEVEL_ONE.value: 1,
        # comfort / temp-2 / level 2
        ScheduleEntryMode.COMFORT.value: 2,
        ScheduleEntryMode.TEMP_2.value: 2,
        ScheduleEntryMode.LEVEL_TWO.value: 2,
        # level 3
        ScheduleEntryMode.LEVEL_THREE.value: 3,
        # level 4
        ScheduleEntryMode.LEVEL_FOUR.value: 4,
    }

    schedule_df: pd.DataFrame

    type: str

    offset: int

    def __init__(self, schedule: Union[Dict, str], type: str, offset: Union[str, int] = None):
        if isinstance(schedule, str):
            schedule = json.loads(schedule)

        self.schedule_df = self.to_dataframe(schedule, offset, type)
        self.type = type
        self.offset = int(offset or 0)

    def to_dataframe(self, schedule: Dict = None, offset: int = None, type: str = "heating"):

        if schedule is None:
            return self.schedule_df

        default_value = 0
        if type == "ventilation":
            default_value = 1

        schedule_df = pd.DataFrame(
            data={
                "timestamp": pd.date_range(
                    start="2020-01-06 00:00:00", end="2020-01-12 23:59:00", freq="T"
                ),
                "value": default_value,
            }
        )

        schedule_df["time"] = schedule_df["timestamp"].dt.time
        schedule_df["weekday"] = schedule_df.timestamp.dt.weekday.map(
            {i: day for i, day in enumerate(self.weekdays)}
        )
        for day, day_schedule in schedule.items():
            for p in sorted(day_schedule, key=lambda x: self.mode_to_value[x["mode"]]):
                mode = p["mode"]

                # start
                start = p["start"]
                if start != "24:00" and start != "00:00":
                    start = datetime.strptime(start, "%H:%M").time()

                else:
                    start = time(0, 0)

                # end
                end = p["end"]
                if end != "24:00" and end != "00:00":
                    end = datetime.strptime(end, "%H:%M").time()

                else:
                    end = time(23, 59)

                flt = (schedule_df["weekday"] == day) & (schedule_df["time"] >= start)
                if end != time(23, 59) and start != end:
                    flt = flt & (schedule_df["time"] < end)

                else:
                    flt = flt & (schedule_df["time"] <= end)

                schedule_df.loc[flt, "value"] = self.mode_to_value[mode]

        # apply offset
        if offset:
            schedule_df = schedule_df.set_index("timestamp")
            schedule_df = schedule_df.shift(periods=-offset, freq="T")
            schedule_df = schedule_df.reset_index()
            schedule_df["weekday"] = schedule_df.timestamp.dt.weekday.map(
                {i: day for i, day in enumerate(self.weekdays)}
            )

        return schedule_df

    #
    # DURATION
    #
    def get_total_duration(self, mode: str = None):
        return {weekday: self.get_duration(weekday, mode) for weekday in self.weekdays}

    def get_duration(self, weekday: Union[str, int, float], mode=None):

        if isinstance(weekday, (int, float)):
            weekday = self.weekdays[int(weekday)]

        daily_schedule = self.schedule_df[self.schedule_df["weekday"] == weekday]["value"]

        if mode:
            return int((daily_schedule == self.mode_to_value[mode]).sum())

        else:
            return int((daily_schedule >= 1).sum())

    #
    # MERGE
    #
    def merge(self, right: "Schedule"):

        left = self

        # check if both schedules have a time offset or not
        if left.offset is None and right.offset is not None:
            raise AttributeError(
                "Merge not possible: left schedule is naive and right schedule has time offset"
            )

        if left.offset is not None and right.offset is None:
            raise AttributeError(
                "Merge not possible: left schedule has time offset and right schedule is naive"
            )

        if left.offset != right.offset:
            raise AttributeError(
                "Merge not possible: left and right schedules have different offsets"
            )

        # check if both schedules are from same type
        if left.type != right.type:
            raise AttributeError(
                "Merge not possible: left and right schedules are of different types"
            )

        left.schedule_df = left.schedule_df.merge(
            right.schedule_df,
            how="inner",
            on=["timestamp", "time", "weekday"],
            suffixes=("_left", "_right"),
        )

        left.schedule_df["value"] = left.schedule_df[["value_left", "value_right"]].max(axis=1)
        left.schedule_df = left.schedule_df.drop(columns=["value_left", "value_right"])
        left.schedule_df = left.schedule_df[["timestamp", "value", "time", "weekday"]]

        return left