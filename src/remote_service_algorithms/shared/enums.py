from enum import Enum, unique


@unique
class WeatherPropertyName(Enum):
    WEATHER_SERVICE_CURRENT_TEMPERATURE = "weather.service/current_temperature"

    WEATHER_SERVICE_APPARENT_TEMPERATURE = "weather.service/apparent_temperature"

    WEATHER_SERVICE_DEW_POINT = "weather.service/dew_point"

    WEATHER_SERVICE_HUMIDITY = "weather.service/humidity"

    WEATHER_SERVICE_CLOUD_COVER = "weather.service/cloud_cover"

    WEATHER_SERVICE_PRESSURE = "weather.service/pressure"

    WEATHER_SERVICE_SEA_LEVEL_PRESSURE = "weather.service/sea_level_pressure"

    WEATHER_SERVICE_WIND_SPEED = "weather.service/wind_speed"

    WEATHER_SERVICE_WIND_GUST_SPEED = "weather.service/wind_gust_speed"

    WEATHER_SERVICE_VISIBILITY = "weather.service/visibility"

    WEATHER_SERVICE_PRECIPITATION = "weather.service/precipitation"

    WEATHER_SERVICE_SNOWFALL = "weather.service/snowfall"

    WEATHER_SERVICE_DIRECT_NORMAL_IRRADIANCE = (
        "weather.service/direct_normal_irradiance"
    )

    WEATHER_SERVICE_GLOBAL_HORIZONTAL_IRRADIANCE = (
        "weather.service/global_horizontal_irradiance"
    )

    WEATHER_SERVICE_DIFFUSE_HORIZONTAL_IRRADIANCE = (
        "weather.service/diffuse_horizontal_irradiance"
    )

    WEATHER_SERVICE_SOLAR_RADIATION = "weather.service/solar_radiation"

    WEATHER_SERVICE_ULTRAVIOLET_LIGHT_INDEX = "weather.service/ultraviolet_light_index"


@unique
class IoTPropertyName(Enum):
    ESS_SENSORS_TEMPERATURE_AMBIENT_VALUE = "ess.sensors.temperature.ambient/value"

    ESS_BATTERY_CAPACITYNOMINAL_VALUE = "ess.battery.capacityNominal/value"

    ESS_STATEOFCHARGE_VALUE = "ess.stateOfCharge/value"

    ESS_BATTERY_ENABLED = "ess.battery/enabled"

    ESS_BATTERY_0_INFORMATION_STATEOFCHARGE = "ess.battery.0.information/stateOfCharge"
    ESS_BATTERY_1_INFORMATION_STATEOFCHARGE = "ess.battery.1.information/stateOfCharge"
    ESS_BATTERY_2_INFORMATION_STATEOFCHARGE = "ess.battery.2.information/stateOfCharge"
    ESS_BATTERY_3_INFORMATION_STATEOFCHARGE = "ess.battery.3.information/stateOfCharge"
    ESS_BATTERY_4_INFORMATION_STATEOFCHARGE = "ess.battery.4.information/stateOfCharge"
    ESS_BATTERY_5_INFORMATION_STATEOFCHARGE = "ess.battery.5.information/stateOfCharge"

    ESS_BATTERY_0_INFORMATION_CAPACITYOFEACHMODULE = (
        "ess.battery.0.information/capacityOfEachModule"
    )
    ESS_BATTERY_1_INFORMATION_CAPACITYOFEACHMODULE = (
        "ess.battery.1.information/capacityOfEachModule"
    )
    ESS_BATTERY_2_INFORMATION_CAPACITYOFEACHMODULE = (
        "ess.battery.2.information/capacityOfEachModule"
    )
    ESS_BATTERY_3_INFORMATION_CAPACITYOFEACHMODULE = (
        "ess.battery.3.information/capacityOfEachModule"
    )
    ESS_BATTERY_4_INFORMATION_CAPACITYOFEACHMODULE = (
        "ess.battery.4.information/capacityOfEachModule"
    )
    ESS_BATTERY_5_INFORMATION_CAPACITYOFEACHMODULE = (
        "ess.battery.5.information/capacityOfEachModule"
    )

    ESS_BATTERY_0_INFORMATION_ENERGYOFMODULE = (
        "ess.battery.0.information/energyOfModule"
    )
    ESS_BATTERY_1_INFORMATION_ENERGYOFMODULE = (
        "ess.battery.1.information/energyOfModule"
    )
    ESS_BATTERY_2_INFORMATION_ENERGYOFMODULE = (
        "ess.battery.2.information/energyOfModule"
    )
    ESS_BATTERY_3_INFORMATION_ENERGYOFMODULE = (
        "ess.battery.3.information/energyOfModule"
    )
    ESS_BATTERY_4_INFORMATION_ENERGYOFMODULE = (
        "ess.battery.4.information/energyOfModule"
    )
    ESS_BATTERY_5_INFORMATION_ENERGYOFMODULE = (
        "ess.battery.5.information/energyOfModule"
    )

    ESS_BATTERY_0_INFORMATION_STATEOFHEALTH = "ess.battery.0.information/stateOfHealth"
    ESS_BATTERY_1_INFORMATION_STATEOFHEALTH = "ess.battery.1.information/stateOfHealth"
    ESS_BATTERY_2_INFORMATION_STATEOFHEALTH = "ess.battery.2.information/stateOfHealth"
    ESS_BATTERY_3_INFORMATION_STATEOFHEALTH = "ess.battery.3.information/stateOfHealth"
    ESS_BATTERY_4_INFORMATION_STATEOFHEALTH = "ess.battery.4.information/stateOfHealth"
    ESS_BATTERY_5_INFORMATION_STATEOFHEALTH = "ess.battery.5.information/stateOfHealth"

    ESS_BATTERY_0_INFORMATION_COUNTCOULOMBCHARGE = (
        "ess.battery.0.information/countCoulombCharge"
    )
    ESS_BATTERY_1_INFORMATION_COUNTCOULOMBCHARGE = (
        "ess.battery.1.information/countCoulombCharge"
    )
    ESS_BATTERY_2_INFORMATION_COUNTCOULOMBCHARGE = (
        "ess.battery.2.information/countCoulombCharge"
    )
    ESS_BATTERY_3_INFORMATION_COUNTCOULOMBCHARGE = (
        "ess.battery.3.information/countCoulombCharge"
    )
    ESS_BATTERY_4_INFORMATION_COUNTCOULOMBCHARGE = (
        "ess.battery.4.information/countCoulombCharge"
    )
    ESS_BATTERY_5_INFORMATION_COUNTCOULOMBCHARGE = (
        "ess.battery.5.information/countCoulombCharge"
    )

    ESS_BATTERY_0_INFORMATION_COUNTCOULOMBDISCHARGE = (
        "ess.battery.0.information/countCoulombDischarge"
    )
    ESS_BATTERY_1_INFORMATION_COUNTCOULOMBDISCHARGE = (
        "ess.battery.1.information/countCoulombDischarge"
    )
    ESS_BATTERY_2_INFORMATION_COUNTCOULOMBDISCHARGE = (
        "ess.battery.2.information/countCoulombDischarge"
    )
    ESS_BATTERY_3_INFORMATION_COUNTCOULOMBDISCHARGE = (
        "ess.battery.3.information/countCoulombDischarge"
    )
    ESS_BATTERY_4_INFORMATION_COUNTCOULOMBDISCHARGE = (
        "ess.battery.4.information/countCoulombDischarge"
    )
    ESS_BATTERY_5_INFORMATION_COUNTCOULOMBDISCHARGE = (
        "ess.battery.5.information/countCoulombDischarge"
    )

    ESS_BATTERY_0_INFORMATION_INTERNALBATTERYRESISTANCE = (
        "ess.battery.0.information/internalBatteryResistance"
    )
    ESS_BATTERY_1_INFORMATION_INTERNALBATTERYRESISTANCE = (
        "ess.battery.1.information/internalBatteryResistance"
    )
    ESS_BATTERY_2_INFORMATION_INTERNALBATTERYRESISTANCE = (
        "ess.battery.2.information/internalBatteryResistance"
    )
    ESS_BATTERY_3_INFORMATION_INTERNALBATTERYRESISTANCE = (
        "ess.battery.3.information/internalBatteryResistance"
    )
    ESS_BATTERY_4_INFORMATION_INTERNALBATTERYRESISTANCE = (
        "ess.battery.4.information/internalBatteryResistance"
    )
    ESS_BATTERY_5_INFORMATION_INTERNALBATTERYRESISTANCE = (
        "ess.battery.5.information/internalBatteryResistance"
    )

    HEATING_CALLFORHEAT_CONFIGURATION_REGULATION_MODE = (
        "heating.callForHeat.configuration.regulation/mode"
    )
    HEATING_CALLFORHEAT_HEATING_CURVE_PRESETS_VALUE = (
        "heating.callForHeat.heating.curve.presets/value"
    )
    HEATING_CALLFORHEAT_HEATING_CURVE_SHIFTVALUE = "heating.callForHeat.heating.curve/shiftValue"
    HEATING_CALLFORHEAT_HEATING_CURVE_SLOPEVALUE = "heating.callForHeat.heating.curve/slopeValue"
    HEATING_CALLFORHEAT_TEMPERATURE_VALUE = "heating.callForHeat.temperature/value"
    HEATING_CALLFORHEAT_ACTIVE = "heating.callForHeat/active"

    HEATING_DHW_SENSORS_TEMPERATURE_OUTLET_VALUE = (
        "heating.dhw.sensors.temperature.outlet/value"
    )

    HEATING_DHW_CHARGING_ACTIVE = "heating.dhw.charging/active"

    HEATING_ERRORS_ACTIVE_ENTRIES = "device.messages.errors.raw/entries"

    DEVICE_MESSAGES_INFO_ENTRIES = "device.messages.info.raw/entries"

    DEVICE_MESSAGES_WARNINGS_RAW_ENTRIES = "device.messages.warnings.raw/entries"

    DEVICE_MESSAGES_STATUS_RAW_ENTRIES = "device.messages.status.raw/entries"

    HEATING_BURNERS_0_MODULATION_VALUE = "heating.burners.0.modulation/value"

    HEATING_BURNERS_0_CURRENT_POWER_VALUE = "heating.burners.0.current.power/value"

    HEATING_BURNERS_1_MODULATION_VALUE = "heating.burners.1.modulation/value"

    HEATING_BURNERS_0_ACTIVE = "heating.burners.0/active"

    HEATING_BURNERS_1_ACTIVE = "heating.burners.1/active"

    HEATING_DEVICE_TIME_OFFSET_VALUE = "heating.device.time.offset/value"

    HEATING_VALVES_DIVERTER_HEATDHW_POSITION = (
        "heating.valves.diverter.heatDhw/position"
    )

    HEATING_VALVES_DIVERTER_FUELCELLDHW_POSITION = (
        "heating.valves.diverter.fuelCellDhw/position"
    )

    HEATING_SECONDARYCIRCUIT_VALVES_FOURTHREEWAY_CURRENT = (
        "heating.secondaryCircuit.valves.fourThreeWay/current"
    )

    HEATING_CIRCUITS_0_HEATING_SCHEDULE_ENTRIES = (
        "heating.circuits.0.heating.schedule/entries"
    )
    HEATING_CIRCUITS_1_HEATING_SCHEDULE_ENTRIES = (
        "heating.circuits.1.heating.schedule/entries"
    )
    HEATING_CIRCUITS_2_HEATING_SCHEDULE_ENTRIES = (
        "heating.circuits.2.heating.schedule/entries"
    )
    HEATING_CIRCUITS_3_HEATING_SCHEDULE_ENTRIES = (
        "heating.circuits.3.heating.schedule/entries"
    )

    HEATING_CIRCUITS_0_ACTIVE = "heating.circuits.0/active"
    HEATING_CIRCUITS_1_ACTIVE = "heating.circuits.1/active"
    HEATING_CIRCUITS_2_ACTIVE = "heating.circuits.2/active"
    HEATING_CIRCUITS_3_ACTIVE = "heating.circuits.3/active"

    HEATING_CIRCUITS_ENABLED = "heating.circuits/enabled"

    HEATING_CIRCUITS_0_SENSORS_TEMPERATURE_SUPPLY_VALUE = (
        "heating.circuits.0.sensors.temperature.supply/value"
    )
    HEATING_CIRCUITS_1_SENSORS_TEMPERATURE_SUPPLY_VALUE = (
        "heating.circuits.1.sensors.temperature.supply/value"
    )
    HEATING_CIRCUITS_2_SENSORS_TEMPERATURE_SUPPLY_VALUE = (
        "heating.circuits.2.sensors.temperature.supply/value"
    )
    HEATING_CIRCUITS_3_SENSORS_TEMPERATURE_SUPPLY_VALUE = (
        "heating.circuits.3.sensors.temperature.supply/value"
    )

    HEATING_CIRCUITS_0_TEMPERATURE_LEVELS_MAXVALUE = (
        "heating.circuits.0.temperature.levels/maxValue"
    )
    HEATING_CIRCUITS_1_TEMPERATURE_LEVELS_MAXVALUE = (
        "heating.circuits.1.temperature.levels/maxValue"
    )
    HEATING_CIRCUITS_2_TEMPERATURE_LEVELS_MAXVALUE = (
        "heating.circuits.2.temperature.levels/maxValue"
    )
    HEATING_CIRCUITS_3_TEMPERATURE_LEVELS_MAXVALUE = (
        "heating.circuits.3.temperature.levels/maxValue"
    )

    HEATING_DHW_OPERATING_MODES_ACTIVE_VALUE = (
        "heating.dhw.operating.modes.active/value"
    )

    HEATING_CONFIGURATION_REGULATION_MODE = "heating.configuration.regulation/mode"

    HEATING_CONFIGURATION_GASTYPE_VALUE = "heating.configuration.gasType/value"

    HEATING_FUELCELL_OPERATING_MODES_ACTIVE_VALUE = (
        "heating.fuelCell.operating.modes.active/value"
    )

    HEATING_CONFIGURATION_COOLING_MODE = "heating.configuration.cooling/mode"

    HEATING_CIRCUITS_0_OPERATING_MODES_ACTIVE_VALUE = (
        "heating.circuits.0.operating.modes.active/value"
    )
    HEATING_CIRCUITS_1_OPERATING_MODES_ACTIVE_VALUE = (
        "heating.circuits.1.operating.modes.active/value"
    )
    HEATING_CIRCUITS_2_OPERATING_MODES_ACTIVE_VALUE = (
        "heating.circuits.2.operating.modes.active/value"
    )
    HEATING_CIRCUITS_3_OPERATING_MODES_ACTIVE_VALUE = (
        "heating.circuits.3.operating.modes.active/value"
    )

    HEATING_CONFIGURATION_MULTIFAMILYHOUSE_ACTIVE = (
        "heating.configuration.multiFamilyHouse/active"
    )

    HEATING_OPERATING_PROGRAMS_HOLIDAY_ACTIVE = (
        "heating.operating.programs.holiday/active"
    )

    HEATING_OPERATING_PROGRAMS_HOLIDAYATHOME_ACTIVE = (
        "heating.operating.programs.holidayAtHome/active"
    )

    HEATING_CIRCUITS_0_OPERATING_PROGRAMS_HOLIDAY_ACTIVE = (
        "heating.circuits.0.operating.programs.holiday/active"
    )
    HEATING_CIRCUITS_1_OPERATING_PROGRAMS_HOLIDAY_ACTIVE = (
        "heating.circuits.1.operating.programs.holiday/active"
    )
    HEATING_CIRCUITS_2_OPERATING_PROGRAMS_HOLIDAY_ACTIVE = (
        "heating.circuits.2.operating.programs.holiday/active"
    )
    HEATING_CIRCUITS_3_OPERATING_PROGRAMS_HOLIDAY_ACTIVE = (
        "heating.circuits.3.operating.programs.holiday/active"
    )

    HEATING_CIRCUITS_0_OPERATING_PROGRAMS_HOLIDAYATHOME_ACTIVE = (
        "heating.circuits.0.operating.programs.holidayAtHome/active"
    )
    HEATING_CIRCUITS_1_OPERATING_PROGRAMS_HOLIDAYATHOME_ACTIVE = (
        "heating.circuits.1.operating.programs.holidayAtHome/active"
    )
    HEATING_CIRCUITS_2_OPERATING_PROGRAMS_HOLIDAYATHOME_ACTIVE = (
        "heating.circuits.2.operating.programs.holidayAtHome/active"
    )
    HEATING_CIRCUITS_3_OPERATING_PROGRAMS_HOLIDAYATHOME_ACTIVE = (
        "heating.circuits.3.operating.programs.holidayAtHome/active"
    )

    HEATING_DHW_TEMPERATURE_HYSTERESIS_VALUE = (
        "heating.dhw.temperature.hysteresis/value"
    )

    HEATING_DHW_ACTIVE = "heating.dhw/active"

    HEATING_DHW_STATUS = "heating.dhw/status"

    HEATING_DHW_TEMPERATURE_MAIN_VALUE = "heating.dhw.temperature.main/value"

    HEATING_DHW_SCHEDULE_ACTIVE = "heating.dhw.schedule/active"

    HEATING_DHW_SCHEDULE_ENTRIES = "heating.dhw.schedule/entries"

    HEATING_CIRCUITS_0_DHW_SCHEDULE_ACTIVE = "heating.circuits.0.dhw.schedule/active"
    HEATING_CIRCUITS_1_DHW_SCHEDULE_ACTIVE = "heating.circuits.1.dhw.schedule/active"
    HEATING_CIRCUITS_2_DHW_SCHEDULE_ACTIVE = "heating.circuits.2.dhw.schedule/active"
    HEATING_CIRCUITS_3_DHW_SCHEDULE_ACTIVE = "heating.circuits.3.dhw.schedule/active"

    HEATING_CIRCUITS_0_DHW_SCHEDULE_ENTRIES = "heating.circuits.0.dhw.schedule/entries"
    HEATING_CIRCUITS_1_DHW_SCHEDULE_ENTRIES = "heating.circuits.1.dhw.schedule/entries"
    HEATING_CIRCUITS_2_DHW_SCHEDULE_ENTRIES = "heating.circuits.2.dhw.schedule/entries"
    HEATING_CIRCUITS_3_DHW_SCHEDULE_ENTRIES = "heating.circuits.3.dhw.schedule/entries"

    HEATING_CIRCUITS_0_HEATING_CURVE_SLOPEVALUE = (
        "heating.circuits.0.heating.curve/slopeValue"
    )
    HEATING_CIRCUITS_1_HEATING_CURVE_SLOPEVALUE = (
        "heating.circuits.1.heating.curve/slopeValue"
    )
    HEATING_CIRCUITS_2_HEATING_CURVE_SLOPEVALUE = (
        "heating.circuits.2.heating.curve/slopeValue"
    )
    HEATING_CIRCUITS_3_HEATING_CURVE_SLOPEVALUE = (
        "heating.circuits.3.heating.curve/slopeValue"
    )

    HEATING_CIRCUITS_0_HEATING_CURVE_SHIFTVALUE = (
        "heating.circuits.0.heating.curve/shiftValue"
    )
    HEATING_CIRCUITS_1_HEATING_CURVE_SHIFTVALUE = (
        "heating.circuits.1.heating.curve/shiftValue"
    )
    HEATING_CIRCUITS_2_HEATING_CURVE_SHIFTVALUE = (
        "heating.circuits.2.heating.curve/shiftValue"
    )
    HEATING_CIRCUITS_3_HEATING_CURVE_SHIFTVALUE = (
        "heating.circuits.3.heating.curve/shiftValue"
    )

    HEATING_CIRCUITS_0_OPERATING_PROGRAMS_NORMAL_TEMPERATUREVALUE = (
        "heating.circuits.0.operating.programs.normal/temperatureValue"
    )
    HEATING_CIRCUITS_1_OPERATING_PROGRAMS_NORMAL_TEMPERATUREVALUE = (
        "heating.circuits.1.operating.programs.normal/temperatureValue"
    )
    HEATING_CIRCUITS_2_OPERATING_PROGRAMS_NORMAL_TEMPERATUREVALUE = (
        "heating.circuits.2.operating.programs.normal/temperatureValue"
    )
    HEATING_CIRCUITS_3_OPERATING_PROGRAMS_NORMAL_TEMPERATUREVALUE = (
        "heating.circuits.3.operating.programs.normal/temperatureValue"
    )

    HEATING_CIRCUITS_0_OPERATING_PROGRAMS_COMFORT_TEMPERATUREVALUE = (
        "heating.circuits.0.operating.programs.comfort/temperatureValue"
    )
    HEATING_CIRCUITS_1_OPERATING_PROGRAMS_COMFORT_TEMPERATUREVALUE = (
        "heating.circuits.1.operating.programs.comfort/temperatureValue"
    )
    HEATING_CIRCUITS_2_OPERATING_PROGRAMS_COMFORT_TEMPERATUREVALUE = (
        "heating.circuits.2.operating.programs.comfort/temperatureValue"
    )
    HEATING_CIRCUITS_3_OPERATING_PROGRAMS_COMFORT_TEMPERATUREVALUE = (
        "heating.circuits.3.operating.programs.comfort/temperatureValue"
    )

    HEATING_CIRCUITS_0_OPERATING_PROGRAMS_REDUCED_TEMPERATUREVALUE = (
        "heating.circuits.0.operating.programs.reduced/temperatureValue"
    )
    HEATING_CIRCUITS_1_OPERATING_PROGRAMS_REDUCED_TEMPERATUREVALUE = (
        "heating.circuits.1.operating.programs.reduced/temperatureValue"
    )
    HEATING_CIRCUITS_2_OPERATING_PROGRAMS_REDUCED_TEMPERATUREVALUE = (
        "heating.circuits.2.operating.programs.reduced/temperatureValue"
    )
    HEATING_CIRCUITS_3_OPERATING_PROGRAMS_REDUCED_TEMPERATUREVALUE = (
        "heating.circuits.3.operating.programs.reduced/temperatureValue"
    )

    HEATING_SENSORS_TEMPERATURE_OUTSIDE_VALUE = (
        "heating.sensors.temperature.outside/value"
    )

    HEATING_BOILER_SENSORS_TEMPERATURE_MAIN_VALUE = (
        "heating.boiler.sensors.temperature.main/value"
    )

    HEATING_BOILER_SENSORS_TEMPERATURE_COMMONSUPPLY_VALUE = (
        "heating.boiler.sensors.temperature.commonSupply/value"
    )

    HEATING_SENSORS_TEMPERATURE_HYDRAULICSEPARATOR_VALUE = (
        "heating.sensors.temperature.hydraulicSeparator/value"
    )

    HEATING_DHW_SENSORS_TEMPERATURE_HOTWATERSTORAGE_VALUE = (
        "heating.dhw.sensors.temperature.hotWaterStorage/value"
    )

    HEATING_DHW_SENSORS_TEMPERATURE_HOTWATERSTORAGE_TOP_VALUE = (
        "heating.dhw.sensors.temperature.hotWaterStorage.top/value"
    )

    HEATING_DHW_SENSORS_TEMPERATURE_HOTWATERSTORAGE_BOTTOM_VALUE = (
        "heating.dhw.sensors.temperature.hotWaterStorage.bottom/value"
    )

    HEATING_GAS_CONSUMPTION_DHW_DAY = "heating.gas.consumption.dhw/day"

    HEATING_GAS_CONSUMPTION_FUELCELL_DAY = "heating.gas.consumption.fuelCell/day"

    HEATING_GAS_CONSUMPTION_HEATING_DAY = "heating.gas.consumption.heating/day"

    HEATING_GAS_CONSUMPTION_TOTAL_DAY = "heating.gas.consumption.total/day"

    HEATING_GAS_CONSUMPTION_DHW_YEAR = "heating.gas.consumption.dhw/year"

    HEATING_GAS_CONSUMPTION_FUELCELL_YEAR = "heating.gas.consumption.fuelCell/year"

    HEATING_GAS_CONSUMPTION_HEATING_YEAR = "heating.gas.consumption.heating/year"

    HEATING_GAS_CONSUMPTION_TOTAL_YEAR = "heating.gas.consumption.total/year"

    HEATING_GAS_CONSUMPTION_SUMMARY_HEATING_CURRENTDAY = (
        "heating.gas.consumption.summary.heating/currentDay"
    )

    HEATING_GAS_CONSUMPTION_SUMMARY_DHW_CURRENTDAY = (
        "heating.gas.consumption.summary.dhw/currentDay"
    )

    HEATING_GAS_CONSUMPTION_SUMMARY_HEATING_CURRENTYEAR = (
        "heating.gas.consumption.summary.heating/currentYear"
    )

    HEATING_GAS_CONSUMPTION_SUMMARY_DHW_CURRENTYEAR = (
        "heating.gas.consumption.summary.dhw/currentYear"
    )

    HEATING_GAS_CONSUMPTION_SUMMARY_HEATING_LASTYEAR = (
        "heating.gas.consumption.summary.heating/lastYear"
    )

    HEATING_GAS_CONSUMPTION_SUMMARY_DHW_LASTYEAR = (
        "heating.gas.consumption.summary.dhw/lastYear"
    )

    HEATING_HEATINGROD_STATUS_OVERALL = "heating.heatingRod.status/overall"

    HEATING_DEVICE_EMERGENCYOPERATION_ACTIVE = (
        "heating.device.emergencyOperation/active"
    )

    HEATING_HEATINGROD_STATUS_LEVEL1 = "heating.heatingRod.status/level1"

    HEATING_HEATINGROD_STATUS_LEVEL2 = "heating.heatingRod.status/level2"

    HEATING_HEATINGROD_STATUS_LEVEL3 = "heating.heatingRod.status/level3"

    HEATING_HEATINGROD_POWER_CONSUMPTION_HEATING_YEAR = (
        "heating.heatingRod.power.consumption.heating/year"
    )

    HEATING_HEATINGROD_POWER_CONSUMPTION_TOTAL_YEAR = (
        "heating.heatingRod.power.consumption.total/year"
    )

    HEATING_HEATINGROD_POWER_CONSUMPTION_DHW_YEAR = (
        "heating.heatingRod.power.consumption.dhw/year"
    )

    HEATING_POWER_PRODUCTION_CURRENT_VALUE = "heating.power.production.current/value"

    HEATING_FUELCELL_STATISTICS_PRODUCTIONSTARTS = (
        "heating.fuelCell.statistics/productionStarts"
    )

    HEATING_FUELCELL_STATISTICS_PRODUCTIONHOURS = (
        "heating.fuelCell.statistics/productionHours"
    )

    HEATING_FUELCELL_SENSORS_TEMPERATURE_RETURN_VALUE = (
        "heating.fuelCell.sensors.temperature.return/value"
    )

    HEATING_FUELCELL_SENSORS_TEMPERATURE_SUPPLY_VALUE = (
        "heating.fuelCell.sensors.temperature.supply/value"
    )

    HEATING_POWER_PRODUCTION_CUMULATIVE_VALUE = (
        "heating.power.production.cumulative/value"
    )

    HEATING_POWER_PURCHASE_CUMULATIVE_VALUE = "heating.power.purchase.cumulative/value"

    HEATING_POWER_SOLD_CUMULATIVE_VALUE = "heating.power.sold.cumulative/value"

    HEATING_FUELCELL_OPERATING_PHASE_VALUE = "heating.fuelCell.operating.phase/value"

    HEATING_FUELCELL_POWER_PRODUCTION_YEAR = "heating.fuelCell.power.production/year"

    HEATING_SENSORS_PRESSURE_SUPPLY_VALUE = "heating.sensors.pressure.supply/value"

    HEATING_SENSORS_VOLUMETRICFLOW_RETURN_VALUE = (
        "heating.sensors.volumetricFlow.allengra/value"
    )

    HEATING_SECONDARYCIRCUIT_SENSORS_TEMPERATURE_SUPPLY_VALUE = (
        "heating.secondaryCircuit.sensors.temperature.supply/value"
    )

    HEATING_COMPRESSOR_ACTIVE = "heating.compressors.0/active"

    HEATING_COMPRESSORS_0_PHASE = "heating.compressors.0/phase"

    HEATING_SECONDARYCIRCUIT_OPERATION_STATE_CURRENTVALUE = (
        "heating.secondaryCircuit.operation.state/currentValue"
    )
    HEATING_SECONDARYCIRCUIT_OPERATION_STATE_TARGETVALUE = (
        "heating.secondaryCircuit.operation.state/targetValue"
    )

    HEATING_BURNERS_0_STATISTICS_STARTS = "heating.burners.0.statistics/starts"
    HEATING_BURNERS_1_STATISTICS_STARTS = "heating.burners.1.statistics/starts"

    HEATING_CIRCUITS_0_TEMPERATURE_VALUE = "heating.circuits.0.temperature/value"
    HEATING_CIRCUITS_1_TEMPERATURE_VALUE = "heating.circuits.1.temperature/value"
    HEATING_CIRCUITS_2_TEMPERATURE_VALUE = "heating.circuits.2.temperature/value"
    HEATING_CIRCUITS_3_TEMPERATURE_VALUE = "heating.circuits.3.temperature/value"

    HEATING_SOLAR_ACTIVE = "heating.solar/active"

    HEATING_SOLAR_SENSORS_TEMPERATURE_DHW_VALUE = (
        "heating.solar.sensors.temperature.dhw/value"
    )

    HEATING_SOLAR_SENSORS_TEMPERATURE_COLLECTOR_VALUE = (
        "heating.solar.sensors.temperature.collector/value"
    )

    HEATING_SOLAR_PUMPS_CIRCUIT_STATUS = "heating.solar.pumps.circuit/status"

    HEATING_BURNERS_0_STATISTICS_HOURS = "heating.burners.0.statistics/hours"
    HEATING_BURNERS_1_STATISTICS_HOURS = "heating.burners.1.statistics/hours"

    HEATING_COMPRESSOR_STATISTICS_STARTS = "heating.compressors.0.statistics/starts"

    HEATING_COMPRESSOR_STATISTICS_HOURS = "heating.compressors.0.statistics/hours"

    HEATING_SENSORS_VALVE_EXPANSION_VALUE = "heating.sensors.valve.0.expansion/value"

    HEATING_SENSORS_VALVE_1_EXPANSION_VALUE = "heating.sensors.valve.1.expansion/value"

    HEATING_SENSORS_TEMPERATURE_RETURN_VALUE = (
        "heating.sensors.temperature.return/value"
    )

    GATEWAY_WIFI_STRENGTH = "gateway.wifi/strength"

    HEATING_DHW_PUMPS_PRIMARY_STATUS = "heating.dhw.pumps.primary/status"

    HEATING_DHW_PUMPS_CIRCULATION_STATUS = "heating.dhw.pumps.circulation/status"

    DEVICE_TIMESERIES_IGNITIONTIMESTEPS_COUNTONE = (
        "device.timeseries.ignitionTimeSteps/countOne"
    )
    DEVICE_TIMESERIES_IGNITIONTIMESTEPS_COUNTTWO = (
        "device.timeseries.ignitionTimeSteps/countTwo"
    )
    DEVICE_TIMESERIES_IGNITIONTIMESTEPS_COUNTTHREE = (
        "device.timeseries.ignitionTimeSteps/countThree"
    )
    DEVICE_TIMESERIES_IGNITIONTIMESTEPS_COUNTFOUR = (
        "device.timeseries.ignitionTimeSteps/countFour"
    )
    DEVICE_TIMESERIES_IGNITIONTIMESTEPS_COUNTFIVE = (
        "device.timeseries.ignitionTimeSteps/countFive"
    )
    DEVICE_TIMESERIES_IGNITIONTIMESTEPS_COUNTSIX = (
        "device.timeseries.ignitionTimeSteps/countSix"
    )
    DEVICE_TIMESERIES_IGNITIONTIMESTEPS_COUNTSEVEN = (
        "device.timeseries.ignitionTimeSteps/countSeven"
    )
    DEVICE_TIMESERIES_IGNITIONTIMESTEPS_COUNTEIGHT = (
        "device.timeseries.ignitionTimeSteps/countEight"
    )
    DEVICE_TIMESERIES_IGNITIONTIMESTEPS_COUNTNINE = (
        "device.timeseries.ignitionTimeSteps/countNine"
    )
    DEVICE_TIMESERIES_IGNITIONTIMESTEPS_COUNTTEN = (
        "device.timeseries.ignitionTimeSteps/countTen"
    )

    HEATING_COMPRESSORS_0_SPEED_CURRENT_VALUE = (
        "heating.compressors.0.speed.current/value"
    )

    HEATING_NOISE_REDUCTION_OPERATING_PROGRAMS_ACTIVE_VALUE = (
        "heating.noise.reduction.operating.programs.active/value"
    )

    HEATING_PRIMARYCIRCUIT_SENSORS_ROTATION_VALUE = (
        "heating.primaryCircuit.sensors.rotation/value"
    )

    HEATING_PRIMARYCIRCUIT_FANS_0_CURRENT_VALUE = (
        "heating.primaryCircuit.fans.0.current/value"
    )

    HEATING_PRIMARYCIRCUIT_FANS_1_CURRENT_VALUE = (
        "heating.primaryCircuit.fans.1.current/value"
    )

    HEATING_PRIMARYCIRCUIT_SENSORS_TEMPERATURE_SUPPLY_VALUE = (
        "heating.primaryCircuit.sensors.temperature.supply/value"
    )

    HEATING_PRIMARYCIRCUIT_SENSORS_TEMPERATURE_RETURN_VALUE = (
        "heating.primaryCircuit.sensors.temperature.return/value"
    )

    HEATING_FLUE_SENSORS_TEMPERATURE_MAIN_VALUE = (
        "heating.flue.sensors.temperature.main/value"
    )

    HEATING_SENSORS_PRESSURE_SUCTIONGAS_VALUE = (
        "heating.sensors.pressure.suctionGas/value"
    )

    HEATING_SENSORS_TEMPERATURE_SUCTIONGAS_VALUE = (
        "heating.sensors.temperature.suctionGas/value"
    )

    HEATING_COMPRESSORS_0_SENSORS_PRESSURE_INLET_VALUE = (
        "heating.compressors.0.sensors.pressure.inlet/value"
    )

    HEATING_SENSORS_PRESSURE_HOTGAS_VALUE = "heating.sensors.pressure.hotGas/value"

    HEATING_SENSORS_TEMPERATURE_HOTGAS_VALUE = (
        "heating.sensors.temperature.hotGas/value"
    )

    HEATING_SENSORS_TEMPERATURE_LIQUIDGAS_VALUE = (
        "heating.sensors.temperature.liquidGas/value"
    )

    HEATING_COMPRESSORS_0_SENSORS_PRESSURE_OUTLET_VALUE = (
        "heating.compressors.0.sensors.pressure.outlet/value"
    )

    HEATING_COMPRESSORS_0_SENSORS_TEMPERATURE_INLET_VALUE = (
        "heating.compressors.0.sensors.temperature.inlet/value"
    )

    HEATING_COMPRESSORS_0_SENSORS_TEMPERATURE_OUTLET_VALUE = (
        "heating.compressors.0.sensors.temperature.outlet/value"
    )

    HEATING_EVAPORATORS_0_SENSORS_TEMPERATURE_LIQUID_VALUE = (
        "heating.evaporators.0.sensors.temperature.liquid/value"
    )

    HEATING_EVAPORATORS_0_SENSORS_TEMPERATURE_OVERHEAT_VALUE = (
        "heating.evaporators.0.sensors.temperature.overheat/value"
    )

    HEATING_ECONOMIZERS_0_SENSORS_TEMPERATURE_LIQUID_VALUE = (
        "heating.economizers.0.sensors.temperature.liquid/value"
    )

    HEATING_PRIMARYCIRCUIT_VALVES_FOURTHREEWAY_ACTIVE = (
        "heating.primaryCircuit.valves.fourWay/active"
    )

    HEATING_PRIMARYCIRCUIT_VALVES_FOURWAY_STATE = (
        "heating.primaryCircuit.valves.fourWay/state"
    )

    HEATING_COMPRESSORS_0_POWER_CONSUMPTION_DHW_YEAR = (
        "heating.compressors.0.power.consumption.dhw/year"
    )

    HEATING_COMPRESSORS_0_POWER_CONSUMPTION_DHW_WEEK = (
        "heating.compressors.0.power.consumption.dhw.week/value"
    )

    HEATING_COMPRESSORS_0_POWER_CONSUMPTION_COOLING_YEAR = (
        "heating.compressors.0.power.consumption.cooling/year"
    )

    HEATING_COMPRESSORS_0_POWER_CONSUMPTION_COOLING_WEEK = (
        "heating.compressors.0.power.consumption.cooling.week/value"
    )

    HEATING_COMPRESSORS_0_POWER_CONSUMPTION_HEATING_YEAR = (
        "heating.compressors.0.power.consumption.heating/year"
    )

    HEATING_COMPRESSORS_0_POWER_CONSUMPTION_HEATING_WEEK = (
        "heating.compressors.0.power.consumption.heating.week/value"
    )

    HEATING_COMPRESSORS_0_POWER_CONSUMPTION_TOTAL_YEAR = (
        "heating.compressors.0.power.consumption.total/year"
    )
    HEATING_CONDENSORS_0_SENSORS_TEMPERATURE_LIQUID_VALUE = (
        "heating.condensors.0.sensors.temperature.liquid/value"
    )
    HEATING_CONFIGURATION_SECONDARYHEATGENERATOR_MODEONERROR_MODE = (
        "heating.configuration.secondaryHeatGenerator.modeOnError/mode"
    )

    HEATING_CIRCUITS_0_OPERATING_PROGRAMS_SCREEDDRYING_ACTIVE = (
        "heating.circuits.0.operating.programs.screedDrying/active"
    )

    HEATING_CIRCUITS_1_OPERATING_PROGRAMS_SCREEDDRYING_ACTIVE = (
        "heating.circuits.1.operating.programs.screedDrying/active"
    )

    HEATING_CIRCUITS_2_OPERATING_PROGRAMS_SCREEDDRYING_ACTIVE = (
        "heating.circuits.2.operating.programs.screedDrying/active"
    )

    HEATING_CIRCUITS_3_OPERATING_PROGRAMS_SCREEDDRYING_ACTIVE = (
        "heating.circuits.3.operating.programs.screedDrying/active"
    )

    HEATING_HEAT_PRODUCTION_COOLING_YEAR = "heating.heat.production.cooling/year"

    HEATING_HEAT_PRODUCTION_DHW_YEAR = "heating.heat.production.dhw/year"

    HEATING_HEAT_PRODUCTION_HEATING_YEAR = "heating.heat.production.heating/year"

    HEATING_HEAT_PRODUCTION_TOTAL_YEAR = "heating.heat.production.total/year"

    HEATING_COMPRESSORS_0_HEAT_PRODUCTION_COOLING_WEEK = (
        "heating.compressors.0.heat.production.cooling.week/value"
    )

    HEATING_COMPRESSORS_0_HEAT_PRODUCTION_DHW_WEEK = (
        "heating.compressors.0.heat.production.dhw.week/value"
    )

    HEATING_COMPRESSORS_0_HEAT_PRODUCTION_HEATING_WEEK = (
        "heating.compressors.0.heat.production.heating.week/value"
    )

    ROOMS_STATUS_ACTIVE = "rooms.status/active"

    HEATING_SECONDARYHEATGENERATOR_TEMPERATURE_CURRENT = (
        "heating.secondaryHeatGenerator.temperature.current/value"
    )

    HEATING_SECONDARYHEATGENERATOR_VALVES_THREEWAY_TARGET = (
        "heating.secondaryHeatGenerator.valves.threeWay/target"
    )

    HEATING_SECONDARYHEATGENERATOR_VALVES_THREEWAY_VALUE = (
        "heating.secondaryHeatGenerator.valves.threeWay/value"
    )

    HEATING_SPF_DHW_VALUE = "heating.spf.dhw/value"
    HEATING_SPF_HEATING_VALUE = "heating.spf.heating/value"
    HEATING_SPF_TOTAL_VALUE = "heating.spf.total/value"

    VENTILATION_BYPASS_POSITION_VALUE = "ventilation.bypass.position/value"

    VENTILATION_SENSORS_TEMPERATURE_EXTRACT_VALUE = (
        "ventilation.sensors.temperature.extract/value"
    )

    VENTILATION_SENSORS_TEMPERATURE_EXHAUST_VALUE = (
        "ventilation.sensors.temperature.exhaust/value"
    )

    VENTILATION_SENSORS_TEMPERATURE_OUTSIDE_VALUE = (
        "ventilation.sensors.temperature.outside/value"
    )

    VENTILATION_SENSORS_TEMPERATURE_SUPPLY_VALUE = (
        "ventilation.sensors.temperature.supply/value"
    )

    VENTILATION_SCHEDULE_ACTIVE = "ventilation.schedule/active"
    VENTILATION_SCHEDULE_ENTRIES = "ventilation.schedule/entries"
    VENTILATION_FAN_SUPPLY_RUNTIME_VALUE = "ventilation.fan.supply.runtime/value"
    VENTILATION_FAN_EXHAUST_RUNTIME_VALUE = "ventilation.fan.exhaust.runtime/value"
    VENTILATION_OPERATING_MODES_ACTIVE_VALUE = (
        "ventilation.operating.modes.active/value"
    )
    VENTILATION_FAN_FILTER_RUNTIME_VALUE_OPERATING_HOURS = (
        "ventilation.filter.runtime/operatingHours"
    )
    VENTILATION_LEVELS_LEVELONE_VOLUMEFLOW = "ventilation.levels.levelOne/volumeFlow"
    VENTILATION_LEVELS_LEVELTWO_VOLUMEFLOW = "ventilation.levels.levelTwo/volumeFlow"
    VENTILATION_LEVELS_LEVELTHREE_VOLUMEFLOW = "ventilation.levels.levelThree/volumeFlow"
    VENTILATION_LEVELS_LEVELFOUR_VOLUMEFLOW = "ventilation.levels.levelFour/volumeFlow"


def get_prop(name: str) -> IoTPropertyName:
    return getattr(IoTPropertyName, name)