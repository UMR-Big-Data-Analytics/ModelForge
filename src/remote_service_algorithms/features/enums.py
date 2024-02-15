from enum import Enum, unique

from src.remote_service_algorithms.shared.consts import CRITICAL_FUEL_CELL_ERRORS, BATTERY_STATE_OF_CHARGE_HISTOGRAMS


@unique
class FeaturesGroup(Enum):
    DATA_QUALITY = "data_quality"

    ELECTRICITY_STORAGE = "electricity_storage"

    DOMESTIC_HOT_WATER = "domestic_hot_water"

    SUPPLY_TEMPERATURE = "supply_temperature"

    CONFIG = "config"

    WEATHER = "weather"

    HYDRAULIC_SEPARATOR = "hydraulic_separator"

    BOILER_ACTIVITY = "boiler_activity"

    HEATING_ROD = "heating_rod"

    FUEL_CELL = "fuel_cell"

    PRESSURE = "pressure"

    SOLAR = "solar"

    WIFI = "wifi"

    IGNITION_TIMES = "ignition_times"

    REFRIGERATION_CIRCUIT = "refrigeration_circuit"

    BURNER_STARTS_AND_HOURS = "burner_starts_and_hours"

    BURNER_STARTS_AT_VARIOUS_VOLUMETRIC_FLOW_LEVELS = (
        "burner_starts_at_various_volumetric_flow_levels"
    )

    COMPRESSOR_STARTS_AND_HOURS = "compressor_starts_and_hours"

    VOLUMETRIC_FLOW = "volumetric_flow"

    INLET_SENSOR = "inlet_sensor"

    ERRORS = "errors"

    SECONDARYHEATGENERATOR_STARTS_AND_HOURS = "shg_starts_and_hours"

    VENTILATION = "ventilation"

    HYBRID_ACTIVITY = "hybrid_activity"

    DEVICE_INFORMATION = "device_information"

    ENERGY_EFFICIENCY = "energy_efficiency"


@unique
class FeatureName(Enum):
    CONFIG_CALL_FOR_HEAT_ACTIVE = "config_call_for_heat_active"
    CONFIG_WEATHER_CONTROLLED_CALL_FOR_HEAT = "config_weather_controlled_call_for_heat"

    HEATING_CURVE_SLOPE_CALL_FOR_HEAT = "heating_curve_slope_call_for_heat"
    HEATING_CURVE_SHIFT_CALL_FOR_HEAT = "heating_curve_shift_call_for_heat"
    OPERATING_PROGRAMS_NORMAL_TEMPERATURE_CALL_FOR_HEAT = (
        "operating_programs_normal_temperature_call_for_heat"
    )
    HEATING_CURVE_PRESETS_CALL_FOR_HEAT = "heating_curve_presets_call_for_heat"

    TARGET_SUPPLY_MINUS_10_CALL_FOR_HEAT = "target_supply_-10_call_for_heat"
    TARGET_SUPPLY_0_CALL_FOR_HEAT = "target_supply_0_call_for_heat"
    TARGET_SUPPLY_10_CALL_FOR_HEAT = "target_supply_10_call_for_heat"
    TARGET_SUPPLY_MINUS_10_FOR_22_CALL_FOR_HEAT = "target_supply_-10_for_22_call_for_heat"
    TARGET_SUPPLY_0_FOR_22_CALL_FOR_HEAT = "target_supply_0_for_22_call_for_heat"
    TARGET_SUPPLY_10_FOR_22_CALL_FOR_HEAT = "target_supply_10_for_22_call_for_heat"

    AMBIENT_TEMPERATURE_VERY_LOW = "ambient_temperature_very_low"
    AMBIENT_TEMPERATURE_LOW = "ambient_temperature_low"
    AMBIENT_TEMPERATURE_NORMAL = "ambient_temperature_normal"
    AMBIENT_TEMPERATURE_HIGH = "ambient_temperature_high"
    AMBIENT_TEMPERATURE_VERY_HIGH = "ambient_temperature_very_high"

    AMBIENT_TEMPERATURE_MIN = "ambient_temperature_min"
    AMBIENT_TEMPERATURE_MAX = "ambient_temperature_max"
    AMBIENT_TEMPERATURE_MEAN = "ambient_temperature_mean"
    AMBIENT_TEMPERATURE_IQM = "ambient_temperature_iqm"
    AMBIENT_TEMPERATURE_STD = "ambient_temperature_std"

    CONFIG_BATTERY_0_ENABLED = "config_battery_0_enabled"
    CONFIG_BATTERY_1_ENABLED = "config_battery_1_enabled"
    CONFIG_BATTERY_2_ENABLED = "config_battery_2_enabled"
    CONFIG_BATTERY_3_ENABLED = "config_battery_3_enabled"
    CONFIG_BATTERY_4_ENABLED = "config_battery_4_enabled"
    CONFIG_BATTERY_5_ENABLED = "config_battery_5_enabled"

    NUMBER_OF_ENABLED_BATTERIES = "number_of_enabled_batteries"
    TOTAL_ENERGY_CAPACITY_OF_ELECTRICITY_STORAGE = "total_energy_capacity_of_electricity_storage"

    STATE_OF_CHARGE_STORAGE_MIN = "state_of_charge_storage_min"
    STATE_OF_CHARGE_BATTERY_0_MIN = "state_of_charge_battery_0_min"
    STATE_OF_CHARGE_BATTERY_1_MIN = "state_of_charge_battery_1_min"
    STATE_OF_CHARGE_BATTERY_2_MIN = "state_of_charge_battery_2_min"
    STATE_OF_CHARGE_BATTERY_3_MIN = "state_of_charge_battery_3_min"
    STATE_OF_CHARGE_BATTERY_4_MIN = "state_of_charge_battery_4_min"
    STATE_OF_CHARGE_BATTERY_5_MIN = "state_of_charge_battery_5_min"

    STATE_OF_CHARGE_STORAGE_MAX = "state_of_charge_storage_max"
    STATE_OF_CHARGE_BATTERY_0_MAX = "state_of_charge_battery_0_max"
    STATE_OF_CHARGE_BATTERY_1_MAX = "state_of_charge_battery_1_max"
    STATE_OF_CHARGE_BATTERY_2_MAX = "state_of_charge_battery_2_max"
    STATE_OF_CHARGE_BATTERY_3_MAX = "state_of_charge_battery_3_max"
    STATE_OF_CHARGE_BATTERY_4_MAX = "state_of_charge_battery_4_max"
    STATE_OF_CHARGE_BATTERY_5_MAX = "state_of_charge_battery_5_max"

    STATE_OF_CHARGE_STORAGE_MEAN = "state_of_charge_storage_mean"
    STATE_OF_CHARGE_BATTERY_0_MEAN = "state_of_charge_battery_0_mean"
    STATE_OF_CHARGE_BATTERY_1_MEAN = "state_of_charge_battery_1_mean"
    STATE_OF_CHARGE_BATTERY_2_MEAN = "state_of_charge_battery_2_mean"
    STATE_OF_CHARGE_BATTERY_3_MEAN = "state_of_charge_battery_3_mean"
    STATE_OF_CHARGE_BATTERY_4_MEAN = "state_of_charge_battery_4_mean"
    STATE_OF_CHARGE_BATTERY_5_MEAN = "state_of_charge_battery_5_mean"

    RESISTANCE_BATTERY_0_MIN = "resistance_battery_0_min"
    RESISTANCE_BATTERY_1_MIN = "resistance_battery_1_min"
    RESISTANCE_BATTERY_2_MIN = "resistance_battery_2_min"
    RESISTANCE_BATTERY_3_MIN = "resistance_battery_3_min"
    RESISTANCE_BATTERY_4_MIN = "resistance_battery_4_min"
    RESISTANCE_BATTERY_5_MIN = "resistance_battery_5_min"

    RESISTANCE_BATTERY_0_MAX = "resistance_battery_0_max"
    RESISTANCE_BATTERY_1_MAX = "resistance_battery_1_max"
    RESISTANCE_BATTERY_2_MAX = "resistance_battery_2_max"
    RESISTANCE_BATTERY_3_MAX = "resistance_battery_3_max"
    RESISTANCE_BATTERY_4_MAX = "resistance_battery_4_max"
    RESISTANCE_BATTERY_5_MAX = "resistance_battery_5_max"

    RESISTANCE_BATTERY_0_MEAN = "resistance_battery_0_mean"
    RESISTANCE_BATTERY_1_MEAN = "resistance_battery_1_mean"
    RESISTANCE_BATTERY_2_MEAN = "resistance_battery_2_mean"
    RESISTANCE_BATTERY_3_MEAN = "resistance_battery_3_mean"
    RESISTANCE_BATTERY_4_MEAN = "resistance_battery_4_mean"
    RESISTANCE_BATTERY_5_MEAN = "resistance_battery_5_mean"

    RESISTANCE_BATTERY_0_IQM = "resistance_battery_0_iqm"
    RESISTANCE_BATTERY_1_IQM = "resistance_battery_1_iqm"
    RESISTANCE_BATTERY_2_IQM = "resistance_battery_2_iqm"
    RESISTANCE_BATTERY_3_IQM = "resistance_battery_3_iqm"
    RESISTANCE_BATTERY_4_IQM = "resistance_battery_4_iqm"
    RESISTANCE_BATTERY_5_IQM = "resistance_battery_5_iqm"

    RESISTANCE_BATTERY_0_STD = "resistance_battery_0_std"
    RESISTANCE_BATTERY_1_STD = "resistance_battery_1_std"
    RESISTANCE_BATTERY_2_STD = "resistance_battery_2_std"
    RESISTANCE_BATTERY_3_STD = "resistance_battery_3_std"
    RESISTANCE_BATTERY_4_STD = "resistance_battery_4_std"
    RESISTANCE_BATTERY_5_STD = "resistance_battery_5_std"

    COUNT_COULOMB_CHARGE_BATTERY_0_CUMULATIVE = (
        "count_coulomb_charge_battery_0_cumulative"
    )
    COUNT_COULOMB_CHARGE_BATTERY_1_CUMULATIVE = (
        "count_coulomb_charge_battery_1_cumulative"
    )
    COUNT_COULOMB_CHARGE_BATTERY_2_CUMULATIVE = (
        "count_coulomb_charge_battery_2_cumulative"
    )
    COUNT_COULOMB_CHARGE_BATTERY_3_CUMULATIVE = (
        "count_coulomb_charge_battery_3_cumulative"
    )
    COUNT_COULOMB_CHARGE_BATTERY_4_CUMULATIVE = (
        "count_coulomb_charge_battery_4_cumulative"
    )
    COUNT_COULOMB_CHARGE_BATTERY_5_CUMULATIVE = (
        "count_coulomb_charge_battery_5_cumulative"
    )

    COUNT_COULOMB_CHARGE_BATTERY_0 = "count_coulomb_charge_battery_0"
    COUNT_COULOMB_CHARGE_BATTERY_1 = "count_coulomb_charge_battery_1"
    COUNT_COULOMB_CHARGE_BATTERY_2 = "count_coulomb_charge_battery_2"
    COUNT_COULOMB_CHARGE_BATTERY_3 = "count_coulomb_charge_battery_3"
    COUNT_COULOMB_CHARGE_BATTERY_4 = "count_coulomb_charge_battery_4"
    COUNT_COULOMB_CHARGE_BATTERY_5 = "count_coulomb_charge_battery_5"

    COUNT_COULOMB_DISCHARGE_BATTERY_0_CUMULATIVE = (
        "count_coulomb_discharge_battery_0_cumulative"
    )
    COUNT_COULOMB_DISCHARGE_BATTERY_1_CUMULATIVE = (
        "count_coulomb_discharge_battery_1_cumulative"
    )
    COUNT_COULOMB_DISCHARGE_BATTERY_2_CUMULATIVE = (
        "count_coulomb_discharge_battery_2_cumulative"
    )
    COUNT_COULOMB_DISCHARGE_BATTERY_3_CUMULATIVE = (
        "count_coulomb_discharge_battery_3_cumulative"
    )
    COUNT_COULOMB_DISCHARGE_BATTERY_4_CUMULATIVE = (
        "count_coulomb_discharge_battery_4_cumulative"
    )
    COUNT_COULOMB_DISCHARGE_BATTERY_5_CUMULATIVE = (
        "count_coulomb_discharge_battery_5_cumulative"
    )

    COUNT_COULOMB_DISCHARGE_BATTERY_0 = "count_coulomb_discharge_battery_0"
    COUNT_COULOMB_DISCHARGE_BATTERY_1 = "count_coulomb_discharge_battery_1"
    COUNT_COULOMB_DISCHARGE_BATTERY_2 = "count_coulomb_discharge_battery_2"
    COUNT_COULOMB_DISCHARGE_BATTERY_3 = "count_coulomb_discharge_battery_3"
    COUNT_COULOMB_DISCHARGE_BATTERY_4 = "count_coulomb_discharge_battery_4"
    COUNT_COULOMB_DISCHARGE_BATTERY_5 = "count_coulomb_discharge_battery_5"

    CAPACITY_OF_MODULE_BATTERY_0 = "capacity_of_module_battery_0"
    CAPACITY_OF_MODULE_BATTERY_1 = "capacity_of_module_battery_1"
    CAPACITY_OF_MODULE_BATTERY_2 = "capacity_of_module_battery_2"
    CAPACITY_OF_MODULE_BATTERY_3 = "capacity_of_module_battery_3"
    CAPACITY_OF_MODULE_BATTERY_4 = "capacity_of_module_battery_4"
    CAPACITY_OF_MODULE_BATTERY_5 = "capacity_of_module_battery_5"

    ENERGY_OF_MODULE_BATTERY_0 = "energy_of_module_battery_0"
    ENERGY_OF_MODULE_BATTERY_1 = "energy_of_module_battery_1"
    ENERGY_OF_MODULE_BATTERY_2 = "energy_of_module_battery_2"
    ENERGY_OF_MODULE_BATTERY_3 = "energy_of_module_battery_3"
    ENERGY_OF_MODULE_BATTERY_4 = "energy_of_module_battery_4"
    ENERGY_OF_MODULE_BATTERY_5 = "energy_of_module_battery_5"

    STATE_OF_HEALTH_BATTERY_0 = "state_of_health_battery_0"
    STATE_OF_HEALTH_BATTERY_1 = "state_of_health_battery_1"
    STATE_OF_HEALTH_BATTERY_2 = "state_of_health_battery_2"
    STATE_OF_HEALTH_BATTERY_3 = "state_of_health_battery_3"
    STATE_OF_HEALTH_BATTERY_4 = "state_of_health_battery_4"
    STATE_OF_HEALTH_BATTERY_5 = "state_of_health_battery_5"

    YEARLY_DHW_SPF_EMBEDDED = "yearly_dhw_spf_embedded"
    YEARLY_HEATING_SPF_EMBEDDED = "yearly_heating_spf_embedded"
    YEARLY_TOTAL_SPF_EMBEDDED = "yearly_total_spf_embedded"

    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_0_MIN = (
        "ventilation_extract_air_temperature_bypass_0_min"
    )
    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_0_10Q = (
        "ventilation_extract_air_temperature_bypass_0_10q"
    )
    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_0_25Q = (
        "ventilation_extract_air_temperature_bypass_0_25q"
    )
    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_0_75Q = (
        "ventilation_extract_air_temperature_bypass_0_75q"
    )
    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_0_90Q = (
        "ventilation_extract_air_temperature_bypass_0_90q"
    )
    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_0_MAX = (
        "ventilation_extract_air_temperature_bypass_0_max"
    )
    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_0_IQM = (
        "ventilation_extract_air_temperature_bypass_0_iqm"
    )

    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_0_MIN = (
        "ventilation_exhaust_air_temperature_bypass_0_min"
    )
    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_0_10Q = (
        "ventilation_exhaust_air_temperature_bypass_0_10q"
    )
    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_0_25Q = (
        "ventilation_exhaust_air_temperature_bypass_0_25q"
    )
    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_0_75Q = (
        "ventilation_exhaust_air_temperature_bypass_0_75q"
    )
    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_0_90Q = (
        "ventilation_exhaust_air_temperature_bypass_0_90q"
    )
    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_0_MAX = (
        "ventilation_exhaust_air_temperature_bypass_0_max"
    )
    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_0_IQM = (
        "ventilation_exhaust_air_temperature_bypass_0_iqm"
    )

    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_0_MIN = (
        "ventilation_outside_air_temperature_bypass_0_min"
    )
    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_0_10Q = (
        "ventilation_outside_air_temperature_bypass_0_10q"
    )
    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_0_25Q = (
        "ventilation_outside_air_temperature_bypass_0_25q"
    )
    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_0_75Q = (
        "ventilation_outside_air_temperature_bypass_0_75q"
    )
    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_0_90Q = (
        "ventilation_outside_air_temperature_bypass_0_90q"
    )
    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_0_MAX = (
        "ventilation_outside_air_temperature_bypass_0_max"
    )
    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_0_IQM = (
        "ventilation_outside_air_temperature_bypass_0_iqm"
    )

    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_0_MIN = (
        "ventilation_supply_air_temperature_bypass_0_min"
    )
    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_0_10Q = (
        "ventilation_supply_air_temperature_bypass_0_10q"
    )
    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_0_25Q = (
        "ventilation_supply_air_temperature_bypass_0_25q"
    )
    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_0_75Q = (
        "ventilation_supply_air_temperature_bypass_0_75q"
    )
    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_0_90Q = (
        "ventilation_supply_air_temperature_bypass_0_90q"
    )
    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_0_MAX = (
        "ventilation_supply_air_temperature_bypass_0_max"
    )
    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_0_IQM = (
        "ventilation_supply_air_temperature_bypass_0_iqm"
    )

    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_100_MIN = (
        "ventilation_extract_air_temperature_bypass_100_min"
    )
    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_100_10Q = (
        "ventilation_extract_air_temperature_bypass_100_10q"
    )
    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_100_25Q = (
        "ventilation_extract_air_temperature_bypass_100_25q"
    )
    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_100_75Q = (
        "ventilation_extract_air_temperature_bypass_100_75q"
    )
    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_100_90Q = (
        "ventilation_extract_air_temperature_bypass_100_90q"
    )
    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_100_MAX = (
        "ventilation_extract_air_temperature_bypass_100_max"
    )
    VENTILATION_EXTRACT_AIR_TEMPERATURE_BYPASS_100_IQM = (
        "ventilation_extract_air_temperature_bypass_100_iqm"
    )

    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_100_MIN = (
        "ventilation_exhaust_air_temperature_bypass_100_min"
    )
    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_100_10Q = (
        "ventilation_exhaust_air_temperature_bypass_100_10q"
    )
    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_100_25Q = (
        "ventilation_exhaust_air_temperature_bypass_100_25q"
    )
    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_100_75Q = (
        "ventilation_exhaust_air_temperature_bypass_100_75q"
    )
    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_100_90Q = (
        "ventilation_exhaust_air_temperature_bypass_100_90q"
    )
    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_100_MAX = (
        "ventilation_exhaust_air_temperature_bypass_100_max"
    )
    VENTILATION_EXHAUST_AIR_TEMPERATURE_BYPASS_100_IQM = (
        "ventilation_exhaust_air_temperature_bypass_100_iqm"
    )

    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_100_MIN = (
        "ventilation_outside_air_temperature_bypass_100_min"
    )
    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_100_10Q = (
        "ventilation_outside_air_temperature_bypass_100_10q"
    )
    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_100_25Q = (
        "ventilation_outside_air_temperature_bypass_100_25q"
    )
    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_100_75Q = (
        "ventilation_outside_air_temperature_bypass_100_75q"
    )
    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_100_90Q = (
        "ventilation_outside_air_temperature_bypass_100_90q"
    )
    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_100_MAX = (
        "ventilation_outside_air_temperature_bypass_100_max"
    )
    VENTILATION_OUTSIDE_AIR_TEMPERATURE_BYPASS_100_IQM = (
        "ventilation_outside_air_temperature_bypass_100_iqm"
    )

    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_100_MIN = (
        "ventilation_supply_air_temperature_bypass_100_min"
    )
    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_100_10Q = (
        "ventilation_supply_air_temperature_bypass_100_10q"
    )
    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_100_25Q = (
        "ventilation_supply_air_temperature_bypass_100_25q"
    )
    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_100_75Q = (
        "ventilation_supply_air_temperature_bypass_100_75q"
    )
    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_100_90Q = (
        "ventilation_supply_air_temperature_bypass_100_90q"
    )
    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_100_MAX = (
        "ventilation_supply_air_temperature_bypass_100_max"
    )
    VENTILATION_SUPPLY_AIR_TEMPERATURE_BYPASS_100_IQM = (
        "ventilation_supply_air_temperature_bypass_100_iqm"
    )

    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_0_MIN = (
        "ventilation_weather_service_outside_bypass_0_min"
    )
    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_0_10Q = (
        "ventilation_weather_service_outside_bypass_0_10q"
    )
    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_0_25Q = (
        "ventilation_weather_service_outside_bypass_0_25q"
    )
    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_0_75Q = (
        "ventilation_weather_service_outside_bypass_0_75q"
    )
    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_0_90Q = (
        "ventilation_weather_service_outside_bypass_0_90q"
    )
    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_0_MAX = (
        "ventilation_weather_service_outside_bypass_0_max"
    )
    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_0_IQM = (
        "ventilation_weather_service_outside_bypass_0_iqm"
    )

    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_100_MIN = (
        "ventilation_weather_service_outside_bypass_100_min"
    )
    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_100_10Q = (
        "ventilation_weather_service_outside_bypass_100_10q"
    )
    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_100_25Q = (
        "ventilation_weather_service_outside_bypass_100_25q"
    )
    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_100_75Q = (
        "ventilation_weather_service_outside_bypass_100_75q"
    )
    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_100_90Q = (
        "ventilation_weather_service_outside_bypass_100_90q"
    )
    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_100_MAX = (
        "ventilation_weather_service_outside_bypass_100_max"
    )
    VENTILATION_WEATHER_SERVICE_OUTSIDE_BYPASS_100_IQM = (
        "ventilation_weather_service_outside_bypass_100_iqm"
    )
    VENTILATION_DURATION_SCHEDULE = "ventilation_duration_schedule"
    VENTILATION_DURATION_ACTUAL = "ventilation_duration_actual"
    VENTILATION_LEVEL_VOLUMEFLOW = "ventilation_level_volumeflow"
    VENTILATION_MODE = "ventilation_mode"
    VENTILATION_ACTIVE = "ventilation_active"
    VENTILATION_STATUS_CHANGED = "ventilation_status_changed"
    VENTILATION_SCHEDULED_VOLUMELEVEL_ONE = "ventilation_scheduled_volumelevel_one"
    VENTILATION_SCHEDULED_VOLUMELEVEL_TWO = "ventilation_scheduled_volumelevel_two"
    VENTILATION_SCHEDULED_VOLUMELEVEL_THREE = "ventilation_scheduled_volumelevel_three"
    VENTILATION_SCHEDULED_VOLUMELEVEL_FOUR = "ventilation_scheduled_volumelevel_four"
    VENTILATION_LEVELS_LEVELONE_VOLUMEFLOW = "ventilation_levels_levelone_volumeflow"
    VENTILATION_LEVELS_LEVELTWO_VOLUMEFLOW = "ventilation_levels_leveltwo_volumeflow"
    VENTILATION_LEVELS_LEVELTHREE_VOLUMEFLOW = "ventilation_levels_levelthree_volumeflow"
    VENTILATION_LEVELS_LEVELFOUR_VOLUMEFLOW = "ventilation_levels_levelfour_volumeflow"
    VENTILATION_EOD_SUPPLY_RUNTIME_VALUE = "ventilation_eod_supply_runtime_value"
    VENTILATION_EOD_EXHAUST_RUNTIME_VALUE = "ventilation_eod_exhaust_runtime_value"

    OUTLET_NON_DHW_MIN = "outlet_non_dhw_min"

    OUTLET_NON_DHW_MAX = "outlet_non_dhw_max"

    OUTLET_NON_DHW_MEAN = "outlet_non_dhw_mean"

    OUTLET_NON_DHW_IQM = "outlet_non_dhw_iqm"

    OUTLET_NON_DHW_STD = "outlet_non_dhw_std"

    OUTLET_DHW_MIN = "outlet_dhw_min"

    OUTLET_DHW_MAX = "outlet_dhw_max"

    OUTLET_DHW_MEAN = "outlet_dhw_mean"

    OUTLET_DHW_IQM = "outlet_dhw_iqm"

    OUTLET_DHW_STD = "outlet_dhw_std"

    ONLINE_SHARE = "online_share"

    ONLINE_GAPS = "online_gaps"

    CONFIG_HEATING_SCHEDULE_DURATION_HC0 = "config_heating_schedule_duration_hc0"

    CONFIG_HEATING_SCHEDULE_DURATION_HC1 = "config_heating_schedule_duration_hc1"

    CONFIG_HEATING_SCHEDULE_DURATION_HC2 = "config_heating_schedule_duration_hc2"

    CONFIG_HEATING_SCHEDULE_DURATION_HC3 = "config_heating_schedule_duration_hc3"

    CONFIG_HEATING_SCHEDULE_DURATION_REDUCED_HC0 = (
        "config_heating_schedule_duration_reduced_hc0"
    )

    CONFIG_HEATING_SCHEDULE_DURATION_REDUCED_HC1 = (
        "config_heating_schedule_duration_reduced_hc1"
    )

    CONFIG_HEATING_SCHEDULE_DURATION_REDUCED_HC2 = (
        "config_heating_schedule_duration_reduced_hc2"
    )

    CONFIG_HEATING_SCHEDULE_DURATION_REDUCED_HC3 = (
        "config_heating_schedule_duration_reduced_hc3"
    )

    CONFIG_HEATING_SCHEDULE_DURATION_NORMAL_HC0 = (
        "config_heating_schedule_duration_normal_hc0"
    )

    CONFIG_HEATING_SCHEDULE_DURATION_NORMAL_HC1 = (
        "config_heating_schedule_duration_normal_hc1"
    )

    CONFIG_HEATING_SCHEDULE_DURATION_NORMAL_HC2 = (
        "config_heating_schedule_duration_normal_hc2"
    )

    CONFIG_HEATING_SCHEDULE_DURATION_NORMAL_HC3 = (
        "config_heating_schedule_duration_normal_hc3"
    )

    CONFIG_HEATING_SCHEDULE_DURATION_COMFORT_HC0 = (
        "config_heating_schedule_duration_comfort_hc0"
    )

    CONFIG_HEATING_SCHEDULE_DURATION_COMFORT_HC1 = (
        "config_heating_schedule_duration_comfort_hc1"
    )

    CONFIG_HEATING_SCHEDULE_DURATION_COMFORT_HC2 = (
        "config_heating_schedule_duration_comfort_hc2"
    )

    CONFIG_HEATING_SCHEDULE_DURATION_COMFORT_HC3 = (
        "config_heating_schedule_duration_comfort_hc3"
    )

    CONFIG_DHW_OPERATING_MODE = "config_dhw_operating_mode"

    CONFIG_GAS_TYPE = "config_gas_type"

    CONFIG_VACATION = "config_vacation"

    CONFIG_HEATING_ACTIVE = "config_heating_active"

    CONFIG_HEATING_ACTIVE_HC0 = "config_heating_active_hc0"

    CONFIG_HEATING_ACTIVE_HC1 = "config_heating_active_hc1"

    CONFIG_HEATING_ACTIVE_HC2 = "config_heating_active_hc2"

    CONFIG_HEATING_ACTIVE_HC3 = "config_heating_active_hc3"

    CONFIG_WEATHER_CONTROLLED_REGULATION = "config_weather_controlled_regulation"

    CONFIG_FUEL_CELL_ACTIVE = "config_fuel_cell_active"

    CONFIG_COOLING_ACTIVE = "config_cooling_active"

    CONFIG_DHW_HYSTERESIS = "config_dhw_hysteresis"

    CONFIG_DHW_ACTIVE = "config_dhw_active"

    CONFIG_DHW_DURATION = "config_dhw_duration"

    CONFIG_DHW_TARGET_TEMPERATURE = "config_dhw_target_temperature"

    CONFIG_TARGET_SUPPLY_MAX_HC0 = "config_target_supply_max_hc0"

    CONFIG_TARGET_SUPPLY_MAX_HC1 = "config_target_supply_max_hc1"

    CONFIG_TARGET_SUPPLY_MAX_HC2 = "config_target_supply_max_hc2"

    CONFIG_TARGET_SUPPLY_MAX_HC3 = "config_target_supply_max_hc3"

    TARGET_SUPPLY_MINUS_10_HC0 = "target_supply_-10_hc0"

    TARGET_SUPPLY_0_HC0 = "target_supply_0_hc0"

    TARGET_SUPPLY_10_HC0 = "target_supply_10_hc0"

    TARGET_SUPPLY_MINUS_10_FOR_22_HC0 = "target_supply_-10_for_22_hc0"

    TARGET_SUPPLY_0_FOR_22_HC0 = "target_supply_0_for_22_hc0"

    TARGET_SUPPLY_10_FOR_22_HC0 = "target_supply_10_for_22_hc0"

    TARGET_SUPPLY_MINUS_10_HC1 = "target_supply_-10_hc1"

    TARGET_SUPPLY_0_HC1 = "target_supply_0_hc1"

    TARGET_SUPPLY_10_HC1 = "target_supply_10_hc1"

    TARGET_SUPPLY_MINUS_10_FOR_22_HC1 = "target_supply_-10_for_22_hc1"

    TARGET_SUPPLY_0_FOR_22_HC1 = "target_supply_0_for_22_hc1"

    TARGET_SUPPLY_10_FOR_22_HC1 = "target_supply_10_for_22_hc1"

    TARGET_SUPPLY_MINUS_10_HC2 = "target_supply_-10_hc2"

    TARGET_SUPPLY_0_HC2 = "target_supply_0_hc2"

    TARGET_SUPPLY_10_HC2 = "target_supply_10_hc2"

    TARGET_SUPPLY_MINUS_10_FOR_22_HC2 = "target_supply_-10_for_22_hc2"

    TARGET_SUPPLY_0_FOR_22_HC2 = "target_supply_0_for_22_hc2"

    TARGET_SUPPLY_10_FOR_22_HC2 = "target_supply_10_for_22_hc2"

    TARGET_SUPPLY_MINUS_10_HC3 = "target_supply_-10_hc3"

    TARGET_SUPPLY_0_HC3 = "target_supply_0_hc3"

    TARGET_SUPPLY_10_HC3 = "target_supply_10_hc3"

    TARGET_SUPPLY_MINUS_10_FOR_22_HC3 = "target_supply_-10_for_22_hc3"

    TARGET_SUPPLY_0_FOR_22_HC3 = "target_supply_0_for_22_hc3"

    TARGET_SUPPLY_10_FOR_22_HC3 = "target_supply_10_for_22_hc3"

    MINIMUM_OUTSIDE_TEMPERATURE = "minimum_outside_temperature"

    MAXIMUM_OUTSIDE_TEMPERATURE = "maximum_outside_temperature"

    AVERAGE_OUTSIDE_TEMPERATURE = "average_outside_temperature"

    DIFFERENCE_OUTSIDE_TEMPERATURE = "difference_outside_temperature"

    MINIMUM_OUTSIDE_TEMPERATURE_WEATHER_SERVICE = (
        "minimum_outside_temperature_weather_service"
    )

    MAXIMUM_OUTSIDE_TEMPERATURE_WEATHER_SERVICE = (
        "maximum_outside_temperature_weather_service"
    )

    AVERAGE_OUTSIDE_TEMPERATURE_WEATHER_SERVICE = (
        "average_outside_temperature_weather_service"
    )

    DIFFERENCE_OUTSIDE_TEMPERATURE_WEATHER_SERVICE = (
        "difference_outside_temperature_weather_service"
    )

    MINIMUM_APPARENT_TEMPERATURE_WEATHER_SERVICE = (
        "minimum_apparent_temperature_weather_service"
    )

    MAXIMUM_APPARENT_TEMPERATURE_WEATHER_SERVICE = (
        "maximum_apparent_temperature_weather_service"
    )

    AVERAGE_APPARENT_TEMPERATURE_WEATHER_SERVICE = (
        "average_apparent_temperature_weather_service"
    )

    DIFFERENCE_APPARENT_TEMPERATURE_WEATHER_SERVICE = (
        "difference_apparent_temperature_weather_service"
    )

    MINIMUM_DEW_POINT_WEATHER_SERVICE = "minimum_dew_point_weather_service"

    MAXIMUM_DEW_POINT_WEATHER_SERVICE = "maximum_dew_point_weather_service"

    AVERAGE_DEW_POINT_WEATHER_SERVICE = "average_dew_point_weather_service"

    DIFFERENCE_DEW_POINT_WEATHER_SERVICE = "difference_dew_point_weather_service"

    MINIMUM_HUMIDITY_WEATHER_SERVICE = "minimum_humidity_weather_service"

    MAXIMUM_HUMIDITY_WEATHER_SERVICE = "maximum_humidity_weather_service"

    AVERAGE_HUMIDITY_WEATHER_SERVICE = "average_humidity_weather_service"

    DIFFERENCE_HUMIDITY_WEATHER_SERVICE = "difference_humidity_weather_service"

    MINIMUM_CLOUD_COVER_WEATHER_SERVICE = "minimum_cloud_cover_weather_service"

    MAXIMUM_CLOUD_COVER_WEATHER_SERVICE = "maximum_cloud_cover_weather_service"

    AVERAGE_CLOUD_COVER_WEATHER_SERVICE = "average_cloud_cover_weather_service"

    DIFFERENCE_CLOUD_COVER_WEATHER_SERVICE = "difference_cloud_cover_weather_service"

    MINIMUM_AIR_PRESSURE_WEATHER_SERVICE = "minimum_air_pressure_weather_service"

    MAXIMUM_AIR_PRESSURE_WEATHER_SERVICE = "maximum_air_pressure_weather_service"

    AVERAGE_AIR_PRESSURE_WEATHER_SERVICE = "average_air_pressure_weather_service"

    DIFFERENCE_AIR_PRESSURE_WEATHER_SERVICE = "difference_air_pressure_weather_service"

    MINIMUM_SEA_LEVEL_PRESSURE_WEATHER_SERVICE = (
        "minimum_sea_level_pressure_weather_service"
    )

    MAXIMUM_SEA_LEVEL_PRESSURE_WEATHER_SERVICE = (
        "maximum_sea_level_pressure_weather_service"
    )

    AVERAGE_SEA_LEVEL_PRESSURE_WEATHER_SERVICE = (
        "average_sea_level_pressure_weather_service"
    )

    DIFFERENCE_SEA_LEVEL_PRESSURE_WEATHER_SERVICE = (
        "difference_sea_level_pressure_weather_service"
    )

    MINIMUM_WIND_SPEED_WEATHER_SERVICE = "minimum_wind_speed_weather_service"

    MAXIMUM_WIND_SPEED_WEATHER_SERVICE = "maximum_wind_speed_weather_service"

    AVERAGE_WIND_SPEED_WEATHER_SERVICE = "average_wind_speed_weather_service"

    DIFFERENCE_WIND_SPEED_WEATHER_SERVICE = "difference_wind_speed_weather_service"

    MINIMUM_WIND_GUST_SPEED_WEATHER_SERVICE = "minimum_wind_gust_speed_weather_service"

    MAXIMUM_WIND_GUST_SPEED_WEATHER_SERVICE = "maximum_wind_gust_speed_weather_service"

    AVERAGE_WIND_GUST_SPEED_WEATHER_SERVICE = "average_wind_gust_speed_weather_service"

    DIFFERENCE_WIND_GUST_SPEED_WEATHER_SERVICE = (
        "difference_wind_gust_speed_weather_service"
    )

    MINIMUM_VISIBILITY_WEATHER_SERVICE = "minimum_visibility_weather_service"

    MAXIMUM_VISIBILITY_WEATHER_SERVICE = "maximum_visibility_weather_service"

    AVERAGE_VISIBILITY_WEATHER_SERVICE = "average_visibility_weather_service"

    DIFFERENCE_VISIBILITY_WEATHER_SERVICE = "difference_visibility_weather_service"

    MINIMUM_PRECIPITATION_WEATHER_SERVICE = "minimum_precipitation_weather_service"

    MAXIMUM_PRECIPITATION_WEATHER_SERVICE = "maximum_precipitation_weather_service"

    AVERAGE_PRECIPITATION_WEATHER_SERVICE = "average_precipitation_weather_service"

    DIFFERENCE_PRECIPITATION_WEATHER_SERVICE = (
        "difference_precipitation_weather_service"
    )

    MINIMUM_SNOWFALL_WEATHER_SERVICE = "minimum_snowfall_weather_service"

    MAXIMUM_SNOWFALL_WEATHER_SERVICE = "maximum_snowfall_weather_service"

    AVERAGE_SNOWFALL_WEATHER_SERVICE = "average_snowfall_weather_service"

    DIFFERENCE_SNOWFALL_WEATHER_SERVICE = "difference_snowfall_weather_service"

    MINIMUM_DIRECT_NORMAL_IRRADIANCE_WEATHER_SERVICE = (
        "minimum_direct_normal_irradiance_weather_service"
    )

    MAXIMUM_DIRECT_NORMAL_IRRADIANCE_WEATHER_SERVICE = (
        "maximum_direct_normal_irradiance_weather_service"
    )

    AVERAGE_DIRECT_NORMAL_IRRADIANCE_WEATHER_SERVICE = (
        "average_direct_normal_irradiance_weather_service"
    )

    DIFFERENCE_DIRECT_NORMAL_IRRADIANCE_WEATHER_SERVICE = (
        "difference_direct_normal_irradiance_weather_service"
    )

    MINIMUM_GLOBAL_HORIZONTAL_IRRADIANCE_WEATHER_SERVICE = (
        "minimum_global_horizontal_irradiance_weather_service"
    )

    MAXIMUM_GLOBAL_HORIZONTAL_IRRADIANCE_WEATHER_SERVICE = (
        "maximum_global_horizontal_irradiance_weather_service"
    )

    AVERAGE_GLOBAL_HORIZONTAL_IRRADIANCE_WEATHER_SERVICE = (
        "average_global_horizontal_irradiance_weather_service"
    )

    DIFFERENCE_GLOBAL_HORIZONTAL_IRRADIANCE_WEATHER_SERVICE = (
        "difference_global_horizontal_irradiance_weather_service"
    )

    MINIMUM_DIFFUSE_HORIZONTAL_IRRADIANCE_WEATHER_SERVICE = (
        "minimum_diffuse_horizontal_irradiance_weather_service"
    )

    MAXIMUM_DIFFUSE_HORIZONTAL_IRRADIANCE_WEATHER_SERVICE = (
        "maximum_diffuse_horizontal_irradiance_weather_service"
    )

    AVERAGE_DIFFUSE_HORIZONTAL_IRRADIANCE_WEATHER_SERVICE = (
        "average_diffuse_horizontal_irradiance_weather_service"
    )

    DIFFERENCE_DIFFUSE_HORIZONTAL_IRRADIANCE_WEATHER_SERVICE = (
        "difference_diffuse_horizontal_irradiance_weather_service"
    )

    MINIMUM_SOLAR_RADIATION_WEATHER_SERVICE = "minimum_solar_radiation_weather_service"

    MAXIMUM_SOLAR_RADIATION_WEATHER_SERVICE = "maximum_solar_radiation_weather_service"

    AVERAGE_SOLAR_RADIATION_WEATHER_SERVICE = "average_solar_radiation_weather_service"

    DIFFERENCE_SOLAR_RADIATION_WEATHER_SERVICE = (
        "difference_solar_radiation_weather_service"
    )

    MINIMUM_ULTRAVIOLET_LIGHT_INDEX_WEATHER_SERVICE = (
        "minimum_ultraviolet_light_index_weather_service"
    )

    MAXIMUM_ULTRAVIOLET_LIGHT_INDEX_WEATHER_SERVICE = (
        "maximum_ultraviolet_light_index_weather_service"
    )

    AVERAGE_ULTRAVIOLET_LIGHT_INDEX_WEATHER_SERVICE = (
        "average_ultraviolet_light_index_weather_service"
    )

    DIFFERENCE_ULTRAVIOLET_LIGHT_INDEX_WEATHER_SERVICE = (
        "difference_ultraviolet_light_index_weather_service"
    )

    HYDRAULIC_SEPARATOR_MIN = "hydraulic_separator_min"

    HYDRAULIC_SEPARATOR_MAX = "hydraulic_separator_max"

    HYDRAULIC_SEPARATOR_IQM = "hydraulic_separator_iqm"

    HEATING_CORR_HYDR_TO_MAIN_TEMP = "heating_corr_hydr_to_main_temp"

    HEATING_MAE_HYDR_TO_MAIN_TEMP = "heating_mae_hydr_to_main_temp"

    HEATING_HYDRAULIC_SEPARATOR_MIN = "heating_hydraulic_separator_min"

    HEATING_HYDRAULIC_SEPARATOR_MAX = "heating_hydraulic_separator_max"

    HEATING_HYDRAULIC_SEPARATOR_IQM = "heating_hydraulic_separator_iqm"

    GAS_CONSUMPTION_DHW_CONTROLLER = "gas_consumption_dhw_controller"

    GAS_CONSUMPTION_FUEL_CELL_CONTROLLER = "gas_consumption_fuel_cell_controller"

    GAS_CONSUMPTION_HEATING_CONTROLLER = "gas_consumption_heating_controller"

    GAS_CONSUMPTION_TOTAL_CONTROLLER = "gas_consumption_total_controller"

    GAS_CONSUMPTION_DHW_CLOUD = "gas_consumption_dhw_cloud"

    GAS_CONSUMPTION_HEATING_CLOUD = "gas_consumption_heating_cloud"

    GAS_CONSUMPTION_TOTAL_CLOUD = "gas_consumption_total_cloud"

    GAS_CONSUMPTION_DHW = "gas_consumption_dhw"

    GAS_CONSUMPTION_FUEL_CELL = "gas_consumption_fuel_cell"

    GAS_CONSUMPTION_HEATING = "gas_consumption_heating"

    GAS_CONSUMPTION_TOTAL = "gas_consumption_total"

    CUMULATIVE_GAS_CONSUMPTION_DHW = "cumulative_gas_consumption_dhw"

    CUMULATIVE_GAS_CONSUMPTION_FUEL_CELL = "cumulative_gas_consumption_fuel_cell"

    CUMULATIVE_GAS_CONSUMPTION_HEATING = "cumulative_gas_consumption_heating"

    CUMULATIVE_GAS_CONSUMPTION_TOTAL = "cumulative_gas_consumption_total"

    EMERGENCY_MODE = "emergency_mode"

    OVERALL_TOTAL_RUNTIME_HEATING_ROD = "overall_total_runtime_heating_rod"

    LEVEL_1_TOTAL_RUNTIME_HEATING_ROD = "level_1_total_runtime_heating_rod"

    LEVEL_2_TOTAL_RUNTIME_HEATING_ROD = "level_2_total_runtime_heating_rod"

    LEVEL_3_TOTAL_RUNTIME_HEATING_ROD = "level_3_total_runtime_heating_rod"

    OVERALL_DHW_RUNTIME_HEATING_ROD = "overall_dhw_runtime_heating_rod"

    LEVEL_1_DHW_RUNTIME_HEATING_ROD = "level_1_dhw_runtime_heating_rod"

    LEVEL_2_DHW_RUNTIME_HEATING_ROD = "level_2_dhw_runtime_heating_rod"

    LEVEL_3_DHW_RUNTIME_HEATING_ROD = "level_3_dhw_runtime_heating_rod"

    OVERALL_HEATING_RUNTIME_HEATING_ROD = "overall_heating_runtime_heating_rod"

    LEVEL_1_HEATING_RUNTIME_HEATING_ROD = "level_1_heating_runtime_heating_rod"

    LEVEL_2_HEATING_RUNTIME_HEATING_ROD = "level_2_heating_runtime_heating_rod"

    LEVEL_3_HEATING_RUNTIME_HEATING_ROD = "level_3_heating_runtime_heating_rod"

    OVERALL_HEATING_BIVALENCE_RUNTIME_HEATING_ROD = (
        "overall_heating_bivalence_runtime_heating_rod"
    )

    LEVEL_1_HEATING_BIVALENCE_RUNTIME_HEATING_ROD = (
        "level_1_heating_bivalence_runtime_heating_rod"
    )

    LEVEL_2_HEATING_BIVALENCE_RUNTIME_HEATING_ROD = (
        "level_2_heating_bivalence_runtime_heating_rod"
    )

    LEVEL_3_HEATING_BIVALENCE_RUNTIME_HEATING_ROD = (
        "level_3_heating_bivalence_runtime_heating_rod"
    )

    OVERALL_HEATING_BIVALENCE_POWER_OFF_RUNTIME_HEATING_ROD = (
        "overall_heating_bivalence_power_off_runtime_heating_rod"
    )

    OVERALL_HEATING_BIVALENCE_NOISE_REDUCED_RUNTIME_HEATING_ROD = (
        "overall_heating_bivalence_noise_reduced_runtime_heating_rod"
    )

    OVERALL_HEATING_BIVALENCE_DEFROST_RUNTIME_HEATING_ROD = (
        "overall_heating_bivalence_defrost_runtime_heating_rod"
    )

    OVERALL_HEATING_BIVALENCE_HIGH_MODULATION_RUNTIME_HEATING_ROD = (
        "overall_heating_bivalence_high_modulation_runtime_heating_rod"
    )

    OVERALL_HEATING_BIVALENCE_PAUSE_RUNTIME_HEATING_ROD = (
        "overall_heating_bivalence_pause_runtime_heating_rod"
    )

    OVERALL_POWER_OFF_RUNTIME_HEATING_ROD = "overall_power_off_runtime_heating_rod"

    LEVEL_1_POWER_OFF_RUNTIME_HEATING_ROD = "level_1_power_off_runtime_heating_rod"

    LEVEL_2_POWER_OFF_RUNTIME_HEATING_ROD = "level_2_power_off_runtime_heating_rod"

    LEVEL_3_POWER_OFF_RUNTIME_HEATING_ROD = "level_3_power_off_runtime_heating_rod"

    OVERALL_NOISE_REDUCED_RUNTIME_HEATING_ROD = (
        "overall_noise_reduced_runtime_heating_rod"
    )

    LEVEL_1_NOISE_REDUCED_RUNTIME_HEATING_ROD = (
        "level_1_noise_reduced_runtime_heating_rod"
    )

    LEVEL_2_NOISE_REDUCED_RUNTIME_HEATING_ROD = (
        "level_2_noise_reduced_runtime_heating_rod"
    )

    LEVEL_3_NOISE_REDUCED_RUNTIME_HEATING_ROD = (
        "level_3_noise_reduced_runtime_heating_rod"
    )

    OVERALL_DEFROST_RUNTIME_HEATING_ROD = "overall_defrost_runtime_heating_rod"

    LEVEL_1_DEFROST_RUNTIME_HEATING_ROD = "level_1_defrost_runtime_heating_rod"

    LEVEL_2_DEFROST_RUNTIME_HEATING_ROD = "level_2_defrost_runtime_heating_rod"

    LEVEL_3_DEFROST_RUNTIME_HEATING_ROD = "level_3_defrost_runtime_heating_rod"

    OVERALL_HIGH_MODULATION_RUNTIME_HEATING_ROD = (
        "overall_high_modulation_runtime_heating_rod"
    )

    LEVEL_1_HIGH_MODULATION_RUNTIME_HEATING_ROD = (
        "level_1_high_modulation_runtime_heating_rod"
    )

    LEVEL_2_HIGH_MODULATION_RUNTIME_HEATING_ROD = (
        "level_2_high_modulation_runtime_heating_rod"
    )

    LEVEL_3_HIGH_MODULATION_RUNTIME_HEATING_ROD = (
        "level_3_high_modulation_runtime_heating_rod"
    )

    OVERALL_PAUSE_RUNTIME_HEATING_ROD = "overall_pause_runtime_heating_rod"

    LEVEL_1_PAUSE_RUNTIME_HEATING_ROD = "level_1_pause_runtime_heating_rod"

    LEVEL_2_PAUSE_RUNTIME_HEATING_ROD = "level_2_pause_runtime_heating_rod"

    LEVEL_3_PAUSE_RUNTIME_HEATING_ROD = "level_3_pause_runtime_heating_rod"

    TOTAL_ENERGY_CONSUMPTION_HEATING_ROD = "total_energy_consumption_heating_rod"

    TOTAL_DHW_ROD_ENERGY_CONSUMPTION_HEATING_ROD = (
        "total_dhw_rod_energy_consumption_heating_rod"
    )

    TOTAL_HEATING_ENERGY_CONSUMPTION_HEATING_ROD = (
        "total_heating_energy_consumption_heating_rod"
    )

    TOTAL_HEATING_BIVALENCE_ENERGY_CONSUMPTION_HEATING_ROD = (
        "total_heating_bivalence_energy_consumption_heating_rod"
    )

    TOTAL_POWER_OFF_ENERGY_CONSUMPTION_HEATING_ROD = (
        "total_power_off_energy_consumption_heating_rod"
    )

    TOTAL_NOISE_REDUCED_ENERGY_CONSUMPTION_HEATING_ROD = (
        "total_noise_reduced_energy_consumption_heating_rod"
    )

    TOTAL_DEFROST_ENERGY_CONSUMPTION_HEATING_ROD = (
        "total_defrost_energy_consumption_heating_rod"
    )

    TOTAL_HIGH_MODULATION_ENERGY_CONSUMPTION_HEATING_ROD = (
        "total_high_modulation_energy_consumption_heating_rod"
    )

    TOTAL_PAUSE_ENERGY_CONSUMPTION_HEATING_ROD = (
        "total_pause_energy_consumption_heating_rod"
    )

    FUEL_CELL_OUTPUT_MAX = "fuel_cell_output_max"

    FUEL_CELL_STARTS_CUMULATIVE = "fuel_cell_starts_cumulative"

    FUEL_CELL_RUNNING_HOURS_CUMULATIVE = "fuel_cell_running_hours_cumulative"

    FUEL_CELL_OUTPUT_SUM_CUMULATIVE = "fuel_cell_output_sum_cumulative"

    FUEL_CELL_PURCHASE_SUM_CUMULATIVE = "fuel_cell_purchase_sum_cumulative"

    FUEL_CELL_SOLD_SUM_CUMULATIVE = "fuel_cell_sold_sum_cumulative"

    FUEL_CELL_STARTS = "fuel_cell_starts"

    FUEL_CELL_RUNNING_HOURS = "fuel_cell_running_hours"

    FUEL_CELL_OUTPUT_SUM = "fuel_cell_output_sum"

    FUEL_CELL_PURCHASE_SUM = "fuel_cell_purchase_sum"

    FUEL_CELL_SOLD_SUM = "fuel_cell_sold_sum"

    FUEL_CELL_OUTPUT_MEAN = "fuel_cell_output_mean"

    FUEL_CELL_NUMBER_POWER_DROPS = "fuel_cell_number_power_drops"

    FUEL_CELL_POWER_DROP_OUTPUT_MIN = "fuel_cell_power_drop_output_min"

    FUEL_CELL_POWER_DROP_OUTPUT_MAX = "fuel_cell_power_drop_output_max"

    FUEL_CELL_POWER_DROP_OUTPUT_MEAN = "fuel_cell_power_drop_output_mean"

    FUEL_CELL_POWER_PRODUCTION = "fuel_cell_power_production"

    FUEL_CELL_CUMULATIVE_POWER_PRODUCTION = "fuel_cell_cumulative_power_production"

    RESTING_PRESSURE_MEAN = "resting_pressure_mean"

    RESTING_PRESSURE_MIN = "resting_pressure_min"

    RESTING_PRESSURE_MAX = "resting_pressure_max"

    RESTING_PRESSURE_STD = "resting_pressure_std"

    MAIN_TOTAL_HEATING_EVENTS = "main_total_heating_events"

    DHW_TOTAL_CHARGING_EVENTS = "dhw_total_charging_events"

    MAIN_GAS_HEATING_EVENTS = "main_gas_heating_events"

    MAIN_GAS_HEATING_EVENT_DURATION = "main_gas_heating_event_duration"

    MAIN_GAS_TEMPERATURE_DIFFERENCE = "main_gas_temperature_difference"

    MAIN_COMPRESSOR_HEATING_EVENTS = "main_compressor_heating_events"

    MAIN_COMPRESSOR_HEATING_EVENT_DURATION = "main_compressor_heating_event_duration"

    MAIN_COMPRESSOR_TEMPERATURE_DIFFERENCE = "main_compressor_temperature_difference"

    MAIN_HEATING_ROD_HEATING_EVENTS = "main_heating_rod_heating_events"

    MAIN_HEATING_ROD_HEATING_EVENT_DURATION = "main_heating_rod_heating_event_duration"

    MAIN_HEATING_ROD_TEMPERATURE_DIFFERENCE = "main_heating_rod_temperature_difference"

    MAIN_SECONDARYHEATGENERATOR_HEATING_EVENTS = (
        "main_secondaryheatgenerator_heating_events"
    )

    MAIN_SECONDARYHEATGENERATOR_HEATING_EVENT_DURATION = (
        "main_secondaryheatgenerator_heating_event_duration"
    )

    MAIN_SECONDARYHEATGENERATOR_TEMPERATURE_DIFFERENCE = (
        "main_secondaryheatgenerator_temperature_difference"
    )

    MAIN_MIXED_HEATING_EVENTS = "main_mixed_heating_events"

    MAIN_MIXED_HEATING_EVENT_DURATION = "main_mixed_heating_event_duration"

    MAIN_MIXED_TEMPERATURE_DIFFERENCE = "main_mixed_temperature_difference"

    MAIN_UNIDENTIFIED_HEATING_EVENTS = "main_unidentified_heating_events"

    MAIN_UNIDENTIFIED_HEATING_EVENT_DURATION = (
        "main_unidentified_heating_event_duration"
    )

    MAIN_UNIDENTIFIED_TEMPERATURE_DIFFERENCE = (
        "main_unidentified_temperature_difference"
    )

    DHW_USAGE_EVENTS = "dhw_usage_events"

    DHW_USAGE_DURATION = "dhw_usage_duration"

    DHW_COOLING_EVENTS = "dhw_cooling_events"

    DHW_COOLING_DURATION = "dhw_cooling_duration"

    DHW_COOLING_RATE = "dhw_cooling_rate"

    DHW_GAS_CHARGING_EVENTS = "dhw_gas_charging_events"

    DHW_GAS_CHARGING_DURATION = "dhw_gas_charging_duration"

    DHW_GAS_CHARGING_DURATION_MAX = "dhw_gas_charging_duration_max"

    DHW_GAS_MODULATION_INTEGRAL = "dhw_gas_modulation_integral"

    DHW_GAS_CHARGING = "dhw_gas_charging"

    DHW_GAS_CHARGING_RATE = "dhw_gas_charging_rate"

    DHW_GAS_CHARGING_EFFICIENCY = "dhw_gas_charging_efficiency"

    DHW_GAS_STARTS = "dhw_gas_starts"

    DHW_SUCCESSFUL_CHARGINGS = "dhw_successful_chargings"

    DHW_NOT_SUCCESSFUL_CHARGINGS = "dhw_not_successful_chargings"

    DHW_CHARGING_START_TEMPERATURE_IQM = "dhw_charging_start_temperature_iqm"

    DHW_COMPRESSOR_CHARGING_EVENTS = "dhw_compressor_charging_events"

    DHW_COMPRESSOR_CHARGING_DURATION = "dhw_compressor_charging_duration"

    DHW_COMPRESSOR_CHARGING_DURATION_MAX = "dhw_compressor_charging_duration_max"

    DHW_COMPRESSOR_MODULATION_INTEGRAL = "dhw_compressor_modulation_integral"

    DHW_COMPRESSOR_CHARGING = "dhw_compressor_charging"

    DHW_COMPRESSOR_CHARGING_RATE = "dhw_compressor_charging_rate"

    DHW_COMPRESSOR_CHARGING_EFFICIENCY = "dhw_compressor_charging_efficiency"

    DHW_COMPRESSOR_STARTS = "dhw_compressor_starts"

    DHW_HEATING_ROD_CHARGING_EVENTS = "dhw_heating_rod_charging_events"

    DHW_HEATING_ROD_CHARGING_DURATION = "dhw_heating_rod_charging_duration"

    DHW_HEATING_ROD_CHARGING_DURATION_MAX = "dhw_heating_rod_charging_duration_max"

    DHW_HEATING_ROD_CHARGING = "dhw_heating_rod_charging"

    DHW_HEATING_ROD_CHARGING_RATE = "dhw_heating_rod_charging_rate"

    DHW_MIXED_CHARGING_EVENTS = "dhw_mixed_charging_events"

    DHW_MIXED_CHARGING_DURATION = "dhw_mixed_charging_duration"

    DHW_MIXED_CHARGING_DURATION_MAX = "dhw_mixed_charging_duration_max"

    DHW_MIXED_CHARGING = "dhw_mixed_charging"

    DHW_MIXED_CHARGING_RATE = "dhw_mixed_charging_rate"

    DHW_SECONDARYHEATGENERATOR_CHARGING_EVENTS = (
        "dhw_secondaryheatgenerator_charging_events"
    )

    DHW_SECONDARYHEATGENERATOR_CHARGING_DURATION = (
        "dhw_secondaryheatgenerator_charging_duration"
    )

    DHW_SECONDARYHEATGENERATOR_CHARGING_DURATION_MAX = (
        "dhw_secondaryheatgenerator_charging_duration_max"
    )

    DHW_SECONDARYHEATGENERATOR_CHARGING = "dhw_secondaryheatgenerator_charging"

    DHW_SECONDARYHEATGENERATOR_CHARGING_RATE = (
        "dhw_secondaryheatgenerator_charging_rate"
    )

    DHW_UNIDENTIFIED_CHARGING_EVENTS = "dhw_unidentified_charging_events"

    DHW_UNIDENTIFIED_CHARGING_DURATION = "dhw_unidentified_charging_duration"

    DHW_UNIDENTIFIED_CHARGING_DURATION_MAX = "dhw_unidentified_charging_duration_max"

    DHW_UNIDENTIFIED_CHARGING = "dhw_unidentified_charging"

    DHW_UNIDENTIFIED_CHARGING_RATE = "dhw_unidentified_charging_rate"

    DHW_TEMPERATURE_MIN = "dhw_temperature_min"

    DHW_TEMPERATURE_MAX = "dhw_temperature_max"

    DHW_TEMPERATURE_MEAN = "dhw_temperature_mean"

    DHW_TEMPERATURE_IQM = "dhw_temperature_iqm"

    DHW_TEMPERATURE_STD = "dhw_temperature_std"

    DHW_DIFFERENCE_TOP_TO_BOTTOM_IQM = "dhw_difference_top_to_bottom_iqm"

    DHW_DIFFERENCE_TOP_TO_BOTTOM_MIN = "dhw_difference_top_to_bottom_min"

    DHW_DIFFERENCE_TOP_TO_BOTTOM_MAX = "dhw_difference_top_to_bottom_max"

    DHW_DIFFERENCE_TOP_TO_BOTTOM_MEAN = "dhw_difference_top_to_bottom_mean"

    DHW_CORRELATION_TOP_TO_BOTTOM = "dhw_correlation_top_to_bottom"

    DHW_DIFFERENCE_TOP_TO_BOTTOM_SUM = "dhw_difference_top_to_bottom_sum"

    DHW_TOP_TEMP_IQM = "dhw_top_temp_iqm"

    DHW_TOP_TEMP_MIN = "dhw_top_temp_min"

    DHW_TOP_TEMP_MAX = "dhw_top_temp_max"

    DHW_TOP_TEMP_MEAN = "dhw_top_temp_mean"

    DHW_BOTTOM_TEMP_IQM = "dhw_bottom_temp_iqm"

    DHW_BOTTOM_TEMP_MIN = "dhw_bottom_temp_min"

    DHW_BOTTOM_TEMP_MAX = "dhw_bottom_temp_max"

    DHW_BOTTOM_TEMP_MEAN = "dhw_bottom_temp_mean"

    RMSE_DHW_BOTTOM_TO_HYDR_SEPARATOR = "rmse_dhw_bottom_to_hydr_separator"

    MIN_TARGET_SUPPLY_HC0 = "min_target_supply_hc0"

    MAX_TARGET_SUPPLY_HC0 = "max_target_supply_hc0"

    MEAN_TARGET_SUPPLY_HC0 = "mean_target_supply_hc0"

    IQM_TARGET_SUPPLY_HC0 = "iqm_target_supply_hc0"

    MIN_TARGET_SUPPLY_HC1 = "min_target_supply_hc1"

    MAX_TARGET_SUPPLY_HC1 = "max_target_supply_hc1"

    MEAN_TARGET_SUPPLY_HC1 = "mean_target_supply_hc1"

    IQM_TARGET_SUPPLY_HC1 = "iqm_target_supply_hc1"

    MIN_TARGET_SUPPLY_HC2 = "min_target_supply_hc2"

    MAX_TARGET_SUPPLY_HC2 = "max_target_supply_hc2"

    MEAN_TARGET_SUPPLY_HC2 = "mean_target_supply_hc2"

    IQM_TARGET_SUPPLY_HC2 = "iqm_target_supply_hc2"

    MIN_TARGET_SUPPLY_HC3 = "min_target_supply_hc3"

    MAX_TARGET_SUPPLY_HC3 = "max_target_supply_hc3"

    MEAN_TARGET_SUPPLY_HC3 = "mean_target_supply_hc3"

    IQM_TARGET_SUPPLY_HC3 = "iqm_target_supply_hc3"

    CORR_COLL_TO_MAIN_DHW = "corr_coll_to_main_dhw"

    CORR_COLL_TO_SOLAR_DHW = "corr_coll_to_solar_dhw"

    DIFF_MAIN_DHW_DAY_TO_NIGHT = "diff_main_dhw_day_to_night"

    DIFF_SOLAR_DHW_DAY_TO_NIGHT = "diff_solar_dhw_day_to_night"

    CHARGING_AT_DAY = "charging_at_day"

    CHARGING_AT_NIGHT = "charging_at_night"

    OUTSIDE_NIVEAU = "outside_niveau"

    SOLAR_DHW_MIN = "solar_dhw_min"

    SOLAR_DHW_MAX = "solar_dhw_max"

    SOLAR_DHW_MEAN = "solar_dhw_mean"

    SOLAR_DHW_IQM = "solar_dhw_iqm"

    SOLAR_DHW_STD = "solar_dhw_std"

    COLLECTOR_MIN = "collector_min"

    COLLECTOR_MAX = "collector_max"

    COLLECTOR_MEAN = "collector_mean"

    COLLECTOR_IQM = "collector_iqm"

    COLLECTOR_STD = "collector_std"

    NUMBER_OF_SOLAR_PUMP_EVENTS = "number_of_solar_pump_events"

    SOLAR_PUMP_DURATION = "solar_pump_duration"

    SHORT_SOLAR_PUMP_EVENTS = "short_solar_pump_events"

    SOLAR_PUMP_EVENTS_MIN_10 = "solar_pump_events_min_10"

    NUMBER_OF_NIGHT_SWITCH_ONS = "number_of_night_switch_ons"

    SOLAR_COLLECTOR_TO_DHW_AREA = "solar_collector_to_dhw_area"

    MEAN_TEMP_DHW_AT_SOLAR_PUMP_STOP = "mean_temp_dhw_at_solar_pump_stop"

    SOLAR_PUMP_STOPS_AT_AFTERNOON = "solar_pump_stops_at_afternoon"

    SOLAR_PUMP_STOPS_WITH_DECREASING_COLLECTOR_TEMP = (
        "solar_pump_stops_with_decreasing_collector_temp"
    )

    SOLAR_PUMP_EVENTS_WITH_DECREASING_DHW_TEMP = (
        "solar_pump_events_with_decreasing_dhw_temp"
    )

    NIGHT_CIRCULATION_SHARE = "night_circulation_share"

    SOLAR_STAGNATION_SHARE = "solar_stagnation_share"

    TOO_LARGE_COLLECTOR_TO_DHW_TEMP_DIFF = "too_large_collector_to_dhw_temp_diff"

    MEAN_DHW_TEMP_WHILE_SOLAR_PUMP_ON = "mean_dhw_temp_while_solar_pump_on"

    WIFI_STRENGTH_MIN = "wifi_strength_min"

    WIFI_STRENGTH_MAX = "wifi_strength_max"

    WIFI_STRENGTH_MEAN = "wifi_strength_mean"

    WIFI_STRENGTH_IQM = "wifi_strength_iqm"

    WIFI_STRENGTH_STD = "wifi_strength_std"

    DHW_PUMP_NUMBER_OF_EVENTS = "dhw_pump_number_of_events"

    DHW_PUMP_DURATION_MEAN = "dhw_pump_duration_mean"

    DHW_PUMP_DURATION_TOTAL = "dhw_pump_duration_total"

    DHW_CIRCULATION_PUMP_NUMBER_OF_EVENTS = "dhw_circulation_pump_number_of_events"

    DHW_CIRCULATION_PUMP_DURATION_MEAN = "dhw_circulation_pump_duration_mean"

    DHW_CIRCULATION_PUMP_DURATION_TOTAL = "dhw_circulation_pump_duration_total"

    SECONDARY_SUPPLY_TEMPERATURE_MIN = "secondary_supply_temperature_min"

    SECONDARY_SUPPLY_TEMPERATURE_MAX = "secondary_supply_temperature_max"

    SECONDARY_SUPPLY_TEMPERATURE_MEAN = "secondary_supply_temperature_mean"

    SECONDARY_SUPPLY_TEMPERATURE_STD = "secondary_supply_temperature_std"

    SECONDARY_RETURN_TEMPERATURE_MIN = "secondary_return_temperature_min"

    SECONDARY_RETURN_TEMPERATURE_MAX = "secondary_return_temperature_max"

    SECONDARY_RETURN_TEMPERATURE_MEAN = "secondary_return_temperature_mean"

    SECONDARY_RETURN_TEMPERATURE_STD = "secondary_return_temperature_std"

    SUPPLY_TEMPERATURE_HC0_MIN = "supply_temperature_hc0_min"

    SUPPLY_TEMPERATURE_HC0_MAX = "supply_temperature_hc0_max"

    SUPPLY_TEMPERATURE_HC0_MEAN = "supply_temperature_hc0_mean"

    SUPPLY_TEMPERATURE_HC0_IQM = "supply_temperature_hc0_iqm"

    SUPPLY_TEMPERATURE_HC0_STD = "supply_temperature_hc0_std"

    SUPPLY_TEMPERATURE_HC1_MIN = "supply_temperature_hc1_min"

    SUPPLY_TEMPERATURE_HC1_MAX = "supply_temperature_hc1_max"

    SUPPLY_TEMPERATURE_HC1_MEAN = "supply_temperature_hc1_mean"

    SUPPLY_TEMPERATURE_HC1_IQM = "supply_temperature_hc1_iqm"

    SUPPLY_TEMPERATURE_HC1_STD = "supply_temperature_hc1_std"

    SUPPLY_TEMPERATURE_HC2_MIN = "supply_temperature_hc2_min"

    SUPPLY_TEMPERATURE_HC2_MAX = "supply_temperature_hc2_max"

    SUPPLY_TEMPERATURE_HC2_MEAN = "supply_temperature_hc2_mean"

    SUPPLY_TEMPERATURE_HC2_IQM = "supply_temperature_hc2_iqm"

    SUPPLY_TEMPERATURE_HC2_STD = "supply_temperature_hc2_std"

    SUPPLY_TEMPERATURE_HC3_MIN = "supply_temperature_hc3_min"

    SUPPLY_TEMPERATURE_HC3_MAX = "supply_temperature_hc3_max"

    SUPPLY_TEMPERATURE_HC3_MEAN = "supply_temperature_hc3_mean"

    SUPPLY_TEMPERATURE_HC3_IQM = "supply_temperature_hc3_iqm"

    SUPPLY_TEMPERATURE_HC3_STD = "supply_temperature_hc3_std"

    SYSTEM_PRESSURE_MIN = "system_pressure_min"

    SYSTEM_PRESSURE_MAX = "system_pressure_max"

    SYSTEM_PRESSURE_MEAN = "system_pressure_mean"

    SYSTEM_PRESSURE_IQM = "system_pressure_iqm"

    SYSTEM_PRESSURE_STD = "system_pressure_std"

    SYSTEM_PRESSURE_Q02 = "system_pressure_q02"

    SYSTEM_PRESSURE_Q98 = "system_pressure_q98"

    MAIN_TEMPERATURE_MIN = "main_temperature_min"

    MAIN_TEMPERATURE_MAX = "main_temperature_max"

    MAIN_TEMPERATURE_MEAN = "main_temperature_mean"

    MAIN_TEMPERATURE_STD = "main_temperature_std"

    MIN_VOLUMETRIC_FLOW_WHILE_GENERATION = "min_volumetric_flow_while_generation"

    MAX_VOLUMETRIC_FLOW_WHILE_GENERATION = "max_volumetric_flow_while_generation"

    MEAN_VOLUMETRIC_FLOW_WHILE_GENERATION = "mean_volumetric_flow_while_generation"

    IQM_VOLUMETRIC_FLOW_WHILE_GENERATION = "iqm_volumetric_flow_while_generation"

    MIN_VOLUMETRIC_FLOW_OUTSIDE_GENERATION = "min_volumetric_flow_outside_generation"

    MAX_VOLUMETRIC_FLOW_OUTSIDE_GENERATION = "max_volumetric_flow_outside_generation"

    MEAN_VOLUMETRIC_FLOW_OUTSIDE_GENERATION = "mean_volumetric_flow_outside_generation"

    IQM_VOLUMETRIC_FLOW_OUTSIDE_GENERATION = "iqm_volumetric_flow_outside_generation"

    MIN_RETURN_TEMP_WHILE_GENERATION = "min_return_temp_while_generation"

    MAX_RETURN_TEMP_WHILE_GENERATION = "max_return_temp_while_generation"

    MEAN_RETURN_TEMP_WHILE_GENERATION = "mean_return_temp_while_generation"

    IQM_RETURN_TEMP_WHILE_GENERATION = "iqm_return_temp_while_generation"

    MIN_SUPPLY_TEMP_WHILE_GENERATION = "min_supply_temp_while_generation"

    MAX_SUPPLY_TEMP_WHILE_GENERATION = "max_supply_temp_while_generation"

    MEAN_SUPPLY_TEMP_WHILE_GENERATION = "mean_supply_temp_while_generation"

    IQM_SUPPLY_TEMP_WHILE_GENERATION = "iqm_supply_temp_while_generation"

    IQM_MAIN_TEMP_WHILE_MAX_RETURN = "iqm_main_temp_while_max_return"

    IQM_SUPPLY_TEMP_WHILE_MAX_RETURN = "iqm_supply_temp_while_max_return"

    MIN_RETURN_TEMP_OUTSIDE_GENERATION = "min_return_temp_outside_generation"

    MAX_RETURN_TEMP_OUTSIDE_GENERATION = "max_return_temp_outside_generation"

    MEAN_RETURN_TEMP_OUTSIDE_GENERATION = "mean_return_temp_outside_generation"

    IQM_RETURN_TEMP_OUTSIDE_GENERATION = "iqm_return_temp_outside_generation"

    DAILY_IGNITION_COUNTER_1 = "daily_ignition_counter_1"

    DAILY_IGNITION_COUNTER_2 = "daily_ignition_counter_2"

    DAILY_IGNITION_COUNTER_3 = "daily_ignition_counter_3"

    DAILY_IGNITION_COUNTER_4 = "daily_ignition_counter_4"

    DAILY_IGNITION_COUNTER_5 = "daily_ignition_counter_5"

    DAILY_IGNITION_COUNTER_6 = "daily_ignition_counter_6"

    DAILY_IGNITION_COUNTER_7 = "daily_ignition_counter_7"

    DAILY_IGNITION_COUNTER_8 = "daily_ignition_counter_8"

    DAILY_IGNITION_COUNTER_9 = "daily_ignition_counter_9"

    DAILY_IGNITION_COUNTER_10 = "daily_ignition_counter_10"

    CUMULATIVE_IGNITION_COUNTER_1 = "cumulative_ignition_counter_1"

    CUMULATIVE_IGNITION_COUNTER_2 = "cumulative_ignition_counter_2"

    CUMULATIVE_IGNITION_COUNTER_3 = "cumulative_ignition_counter_3"

    CUMULATIVE_IGNITION_COUNTER_4 = "cumulative_ignition_counter_4"

    CUMULATIVE_IGNITION_COUNTER_5 = "cumulative_ignition_counter_5"

    CUMULATIVE_IGNITION_COUNTER_6 = "cumulative_ignition_counter_6"

    CUMULATIVE_IGNITION_COUNTER_7 = "cumulative_ignition_counter_7"

    CUMULATIVE_IGNITION_COUNTER_8 = "cumulative_ignition_counter_8"

    CUMULATIVE_IGNITION_COUNTER_9 = "cumulative_ignition_counter_9"

    CUMULATIVE_IGNITION_COUNTER_10 = "cumulative_ignition_counter_10"

    SHORT_IGNITION_TIMES = "short_ignition_times"

    MEDIUM_IGNITION_TIMES = "medium_ignition_times"

    LONG_IGNITION_TIMES = "long_ignition_times"

    NUMBER_OF_COOLING_EVENTS = "number_of_cooling_events"

    COOLING_DURATION_SUM = "cooling_duration_sum"

    COOLING_DURATION_IQM = "cooling_duration_iqm"

    SUPPLY_TEMPERATURE_COOLING_START_IQM = "supply_temperature_cooling_start_iqm"

    SUPPLY_TEMPERATURE_COOLING_END_IQM = "supply_temperature_cooling_end_iqm"

    SUPPLY_TEMPERATURE_COOLING_DIFF_IQM = "supply_temperature_cooling_diff_iqm"

    SUPPLY_OUTLET_DAILY_IQM = "supply_outlet_daily_iqm"

    DHW_OUTLET_DAILY_IQM = "dhw_outlet_daily_iqm"

    DHW_SUPPLY_DAILY_IQM = "dhw_supply_daily_iqm"

    NUMBER_OF_DEFROSTS = "number_of_defrosts"

    DEFROST_DURATION_IQM = "defrost_duration_iqm"

    DIFF_REFRIGERANT_SUPPLY_TO_RETURN_BEFORE_DEFROST_IQM = (
        "diff_refrigerant_supply_to_return_before_defrost_iqm"
    )

    DIFF_REFRIGERANT_SUPPLY_TO_RETURN_AFTER_DEFROST_IQM = (
        "diff_refrigerant_supply_to_return_after_defrost_iqm"
    )

    NUMBER_OF_DEFROST_PREPARATIONS = "number_of_defrost_preparations"

    DEFROST_PREPARATION_DURATION_IQM = "defrost_preparation_duration_iqm"

    CUMULATIVE_BURNER_STARTS = "cumulative_burner_starts"

    CUMULATIVE_BURNER_HOURS = "cumulative_burner_hours"

    BURNER_STARTS = "burner_starts"

    BURNER_HOURS = "burner_hours"

    BURNER_STARTS_PER_HOUR = "burner_starts_per_hour"

    BURNER_HOURS_WHILE_0_TO_10_MODULATION = "burner_hours_while_0_to_10_modulation"
    BURNER_HOURS_WHILE_10_TO_20_MODULATION = "burner_hours_while_10_to_20_modulation"
    BURNER_HOURS_WHILE_20_TO_30_MODULATION = "burner_hours_while_20_to_30_modulation"
    BURNER_HOURS_WHILE_30_TO_40_MODULATION = "burner_hours_while_30_to_40_modulation"
    BURNER_HOURS_WHILE_40_TO_50_MODULATION = "burner_hours_while_40_to_50_modulation"
    BURNER_HOURS_WHILE_50_TO_60_MODULATION = "burner_hours_while_50_to_60_modulation"
    BURNER_HOURS_WHILE_60_TO_70_MODULATION = "burner_hours_while_60_to_70_modulation"
    BURNER_HOURS_WHILE_70_TO_80_MODULATION = "burner_hours_while_70_to_80_modulation"
    BURNER_HOURS_WHILE_80_TO_90_MODULATION = "burner_hours_while_80_to_90_modulation"
    BURNER_HOURS_WHILE_90_TO_100_MODULATION = "burner_hours_while_90_to_100_modulation"
    BURNER_HOURS_WHILE_100_TO_110_MODULATION = "burner_hours_while_100_to_110_modulation"
    BURNER_HOURS_WHILE_110_TO_120_MODULATION = "burner_hours_while_110_to_120_modulation"
    BURNER_HOURS_WHILE_120_TO_130_MODULATION = "burner_hours_while_120_to_130_modulation"
    BURNER_HOURS_WHILE_130_TO_140_MODULATION = "burner_hours_while_130_to_140_modulation"
    BURNER_HOURS_WHILE_140_TO_150_MODULATION = "burner_hours_while_140_to_150_modulation"
    BURNER_HOURS_WHILE_150_TO_160_MODULATION = "burner_hours_while_150_to_160_modulation"
    BURNER_HOURS_WHILE_160_TO_170_MODULATION = "burner_hours_while_160_to_170_modulation"
    BURNER_HOURS_WHILE_170_TO_180_MODULATION = "burner_hours_while_170_to_180_modulation"
    BURNER_HOURS_WHILE_180_TO_190_MODULATION = "burner_hours_while_180_to_190_modulation"
    BURNER_HOURS_WHILE_190_TO_200_MODULATION = "burner_hours_while_190_to_200_modulation"

    CUMULATIVE_BURNER_1_STARTS = "cumulative_burner_1_starts"

    CUMULATIVE_BURNER_1_HOURS = "cumulative_burner_1_hours"

    BURNER_1_STARTS = "burner_1_starts"

    BURNER_1_HOURS = "burner_1_hours"

    BURNER_1_STARTS_PER_HOUR = "burner_1_starts_per_hour"

    CUMULATIVE_COMPRESSOR_STARTS = "cumulative_compressor_starts"

    CUMULATIVE_COMPRESSOR_HOURS = "cumulative_compressor_hours"

    COMPRESSOR_STARTS = "compressor_starts"

    COMPRESSOR_HOURS = "compressor_hours"

    COMPRESSOR_STARTS_PER_HOUR = "compressor_starts_per_hour"

    MAIN_TEMPERATURE_HEATING_IQM = "main_temperature_heating_iqm"

    MAIN_TEMPERATURE_HEATING_MIN = "main_temperature_heating_min"

    MAIN_TEMPERATURE_HEATING_MAX = "main_temperature_heating_max"

    MODULATION_HEATING_IQM = "modulation_heating_iqm"

    MODULATION_HEATING_MIN = "modulation_heating_min"

    MODULATION_HEATING_MAX = "modulation_heating_max"

    FLUE_TEMPERATURE_HEATING_IQM = "flue_temperature_heating_iqm"

    FLUE_TEMPERATURE_HEATING_MIN = "flue_temperature_heating_min"

    FLUE_TEMPERATURE_HEATING_MAX = "flue_temperature_heating_max"

    MAIN_TEMPERATURE_DHW_IQM = "main_temperature_dhw_iqm"

    MAIN_TEMPERATURE_DHW_MIN = "main_temperature_dhw_min"

    MAIN_TEMPERATURE_DHW_MAX = "main_temperature_dhw_max"

    MODULATION_DHW_IQM = "modulation_dhw_iqm"

    MODULATION_DHW_MIN = "modulation_dhw_min"

    MODULATION_DHW_MAX = "modulation_dhw_max"

    FLUE_TEMPERATURE_DHW_IQM = "flue_temperature_dhw_iqm"

    FLUE_TEMPERATURE_DHW_MIN = "flue_temperature_dhw_min"

    FLUE_TEMPERATURE_DHW_MAX = "flue_temperature_dhw_max"

    FLUE_TEMPERATURE_MIN = "flue_temperature_min"

    FLUE_TEMPERATURE_MAX = "flue_temperature_max"

    FLUE_TEMPERATURE_MEAN = "flue_temperature_mean"

    FLUE_TEMPERATURE_STD = "flue_temperature_std"

    MODULATION_HEATING_SIMPLE_SUM = "modulation_heating_simple_sum"

    MODULATION_DHW_SIMPLE_SUM = "modulation_dhw_simple_sum"

    VOLUMETRIC_FLOW_HEATING_MIN = "volumetric_flow_heating_min"

    VOLUMETRIC_FLOW_HEATING_MAX = "volumetric_flow_heating_max"

    VOLUMETRIC_FLOW_HEATING_MEAN = "volumetric_flow_heating_mean"

    VOLUMETRIC_FLOW_HEATING_IQM = "volumetric_flow_heating_iqm"

    VOLUMETRIC_FLOW_HEATING_STD = "volumetric_flow_heating_std"

    VOLUMETRIC_FLOW_DHW_MIN = "volumetric_flow_dhw_min"

    VOLUMETRIC_FLOW_DHW_MAX = "volumetric_flow_dhw_max"

    VOLUMETRIC_FLOW_DHW_MEAN = "volumetric_flow_dhw_mean"

    VOLUMETRIC_FLOW_DHW_IQM = "volumetric_flow_dhw_iqm"

    VOLUMETRIC_FLOW_DHW_STD = "volumetric_flow_dhw_std"

    VOLUMETRIC_FLOW_DEFROST_MIN = "volumetric_flow_defrost_min"

    VOLUMETRIC_FLOW_DEFROST_MAX = "volumetric_flow_defrost_max"

    VOLUMETRIC_FLOW_DEFROST_MEAN = "volumetric_flow_defrost_mean"

    VOLUMETRIC_FLOW_DEFROST_IQM = "volumetric_flow_defrost_iqm"

    VOLUMETRIC_FLOW_DEFROST_STD = "volumetric_flow_defrost_std"

    VOLUMETRIC_FLOW_TOTAL_MIN = "volumetric_flow_total_min"

    VOLUMETRIC_FLOW_TOTAL_MAX = "volumetric_flow_total_max"

    VOLUMETRIC_FLOW_TOTAL_MEAN = "volumetric_flow_total_mean"

    VOLUMETRIC_FLOW_TOTAL_IQM = "volumetric_flow_total_iqm"

    VOLUMETRIC_FLOW_TOTAL_STD = "volumetric_flow_total_std"

    COMPRESSOR_MODULATION_HEATING_IQM = "compressor_modulation_heating_iqm"

    COMPRESSOR_MODULATION_HEATING_MAX = "compressor_modulation_heating_max"

    COMPRESSOR_MODULATION_DHW_IQM = "compressor_modulation_dhw_iqm"

    COMPRESSOR_MODULATION_DHW_MAX = "compressor_modulation_dhw_max"

    COMPRESSOR_MODULATION_COOLING_IQM = "compressor_modulation_cooling_iqm"

    COMPRESSOR_MODULATION_COOLING_MAX = "compressor_modulation_cooling_max"

    COMPRESSOR_MODULATION_DEFROST_IQM = "compressor_modulation_defrost_iqm"

    COMPRESSOR_MODULATION_DEFROST_MAX = "compressor_modulation_defrost_max"

    COMPRESSOR_MODULATION_ACTIVE_IQM = "compressor_modulation_active_iqm"

    COMPRESSOR_MODULATION_ACTIVE_MIN = "compressor_modulation_active_min"

    COMPRESSOR_MODULATION_ACTIVE_MAX = "compressor_modulation_active_max"

    ODU_FAN_ROTATION_IQM = "odu_fan_rotation_iqm"

    ODU_FAN_ROTATION_MIN = "odu_fan_rotation_min"

    ODU_FAN_ROTATION_MAX = "odu_fan_rotation_max"

    ODU_FAN_1_ROTATION_TARGET_IQM = "odu_fan_1_rotation_target_iqm"

    ODU_FAN_1_ROTATION_TARGET_MIN = "odu_fan_1_rotation_target_min"

    ODU_FAN_1_ROTATION_TARGET_MAX = "odu_fan_1_rotation_target_max"

    ODU_FAN_2_ROTATION_TARGET_IQM = "odu_fan_2_rotation_target_iqm"

    ODU_FAN_2_ROTATION_TARGET_MIN = "odu_fan_2_rotation_target_min"

    ODU_FAN_2_ROTATION_TARGET_MAX = "odu_fan_2_rotation_target_max"

    REFRIGERANT_SUPPLY_IQM = "refrigerant_supply_iqm"

    REFRIGERANT_SUPPLY_MIN = "refrigerant_supply_min"

    REFRIGERANT_SUPPLY_MAX = "refrigerant_supply_max"

    REFRIGERANT_SUPPLY_ACTIVE_IQM = "refrigerant_supply_active_iqm"

    REFRIGERANT_SUPPLY_ACTIVE_MIN = "refrigerant_supply_active_min"

    REFRIGERANT_SUPPLY_ACTIVE_MAX = "refrigerant_supply_active_max"

    REFRIGERANT_RETURN_IQM = "refrigerant_return_iqm"

    REFRIGERANT_RETURN_MIN = "refrigerant_return_min"

    REFRIGERANT_RETURN_MAX = "refrigerant_return_max"

    REFRIGERANT_RETURN_ACTIVE_IQM = "refrigerant_return_active_iqm"

    REFRIGERANT_RETURN_ACTIVE_MIN = "refrigerant_return_active_min"

    REFRIGERANT_RETURN_ACTIVE_MAX = "refrigerant_return_active_max"

    EXPANSION_VALVE_IQM = "expansion_valve_iqm"

    EXPANSION_VALVE_MIN = "expansion_valve_min"

    EXPANSION_VALVE_MAX = "expansion_valve_max"

    EXPANSION_VALVE_2_IQM = "expansion_valve_2_iqm"

    EXPANSION_VALVE_2_MIN = "expansion_valve_2_min"

    EXPANSION_VALVE_2_MAX = "expansion_valve_2_max"

    SUCTION_GAS_PRESSURE_IQM = "suction_gas_pressure_iqm"

    SUCTION_GAS_PRESSURE_MIN = "suction_gas_pressure_min"

    SUCTION_GAS_PRESSURE_MAX = "suction_gas_pressure_max"

    SUCTION_GAS_PRESSURE_ACTIVE_IQM = "suction_gas_pressure_active_iqm"

    SUCTION_GAS_PRESSURE_ACTIVE_MIN = "suction_gas_pressure_active_min"

    SUCTION_GAS_PRESSURE_ACTIVE_MAX = "suction_gas_pressure_active_max"

    SUCTION_GAS_PRESSURE_E3_IQM = "suction_gas_pressure_e3_iqm"

    SUCTION_GAS_PRESSURE_E3_MIN = "suction_gas_pressure_e3_min"

    SUCTION_GAS_PRESSURE_E3_MAX = "suction_gas_pressure_e3_max"

    SUCTION_GAS_PRESSURE_E3_ACTIVE_IQM = "suction_gas_pressure_e3_active_iqm"

    SUCTION_GAS_PRESSURE_E3_ACTIVE_MIN = "suction_gas_pressure_e3_active_min"

    SUCTION_GAS_PRESSURE_E3_ACTIVE_MAX = "suction_gas_pressure_e3_active_max"

    HOT_GAS_PRESSURE_IQM = "hot_gas_pressure_iqm"

    HOT_GAS_PRESSURE_MIN = "hot_gas_pressure_min"

    HOT_GAS_PRESSURE_MAX = "hot_gas_pressure_max"

    HOT_GAS_PRESSURE_ACTIVE_IQM = "hot_gas_pressure_active_iqm"

    HOT_GAS_PRESSURE_ACTIVE_MIN = "hot_gas_pressure_active_min"

    HOT_GAS_PRESSURE_ACTIVE_MAX = "hot_gas_pressure_active_max"

    HOT_GAS_PRESSURE_E3_IQM = "hot_gas_pressure_e3_iqm"

    HOT_GAS_PRESSURE_E3_MIN = "hot_gas_pressure_e3_min"

    HOT_GAS_PRESSURE_E3_MAX = "hot_gas_pressure_e3_max"

    HOT_GAS_PRESSURE_E3_ACTIVE_IQM = "hot_gas_pressure_e3_active_iqm"

    HOT_GAS_PRESSURE_E3_ACTIVE_MIN = "hot_gas_pressure_e3_active_min"

    HOT_GAS_PRESSURE_E3_ACTIVE_MAX = "hot_gas_pressure_e3_active_max"

    SUCTION_GAS_TEMPERATURE_IQM = "suction_gas_temperature_iqm"

    SUCTION_GAS_TEMPERATURE_MIN = "suction_gas_temperature_min"

    SUCTION_GAS_TEMPERATURE_MAX = "suction_gas_temperature_max"

    SUCTION_GAS_TEMPERATURE_ACTIVE_IQM = "suction_gas_temperature_active_iqm"

    SUCTION_GAS_TEMPERATURE_ACTIVE_MIN = "suction_gas_temperature_active_min"

    SUCTION_GAS_TEMPERATURE_ACTIVE_MAX = "suction_gas_temperature_active_max"

    HOT_GAS_TEMPERATURE_IQM = "hot_gas_temperature_iqm"

    HOT_GAS_TEMPERATURE_MIN = "hot_gas_temperature_min"

    HOT_GAS_TEMPERATURE_MAX = "hot_gas_temperature_max"

    HOT_GAS_TEMPERATURE_ACTIVE_IQM = "hot_gas_temperature_active_iqm"

    HOT_GAS_TEMPERATURE_ACTIVE_MIN = "hot_gas_temperature_active_min"

    HOT_GAS_TEMPERATURE_ACTIVE_MAX = "hot_gas_temperature_active_max"

    EVAPORATION_TEMPERATURE_IQM = "evaporation_temperature_iqm"

    EVAPORATION_TEMPERATURE_MIN = "evaporation_temperature_min"

    EVAPORATION_TEMPERATURE_MAX = "evaporation_temperature_max"

    EVAPORATION_TEMPERATURE_ACTIVE_IQM = "evaporation_temperature_active_iqm"

    EVAPORATION_TEMPERATURE_ACTIVE_MIN = "evaporation_temperature_active_min"

    EVAPORATION_TEMPERATURE_ACTIVE_MAX = "evaporation_temperature_active_max"

    ECONOMIZER_TEMPERATURE_IQM = "economizer_temperature_iqm"

    ECONOMIZER_TEMPERATURE_MIN = "economizer_temperature_min"

    ECONOMIZER_TEMPERATURE_MAX = "economizer_temperature_max"

    ECONOMIZER_TEMPERATURE_ACTIVE_IQM = "economizer_temperature_active_iqm"

    ECONOMIZER_TEMPERATURE_ACTIVE_MIN = "economizer_temperature_active_min"

    ECONOMIZER_TEMPERATURE_ACTIVE_MAX = "economizer_temperature_active_max"

    CONDENSATION_TEMPERATURE_IQM = "condensation_temperature_iqm"

    CONDENSATION_TEMPERATURE_MIN = "condensation_temperature_min"

    CONDENSATION_TEMPERATURE_MAX = "condensation_temperature_max"

    CONDENSATION_TEMPERATURE_ACTIVE_IQM = "condensation_temperature_active_iqm"

    CONDENSATION_TEMPERATURE_ACTIVE_MIN = "condensation_temperature_active_min"

    CONDENSATION_TEMPERATURE_ACTIVE_MAX = "condensation_temperature_active_max"

    REFRIGERANT_LEAKAGE_METRIC_HEATING = "refrigerant_leakage_metric_heating"

    REFRIGERANT_SUPPLY_STROKE_HEATING = "refrigerant_supply_stroke_heating"

    REFRIGERANT_LEAKAGE_METRIC_DHW = "refrigerant_leakage_metric_dhw"

    REFRIGERANT_SUPPLY_STROKE_DHW = "refrigerant_supply_stroke_dhw"

    REFRIGERANT_LEAKAGE_METRIC_COOLING = "refrigerant_leakage_metric_cooling"

    REFRIGERANT_SUPPLY_STROKE_COOLING = "refrigerant_supply_stroke_cooling"

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_1_ERROR_RATE_HEATING = (
        "refrigerant_temperature_sensor_check_1_heating"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_1_ERROR_RATE_STANDBY = (
        "refrigerant_temperature_sensor_check_1_standby"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_1_ERROR_RATE_DHW_CHARGING = (
        "refrigerant_temperature_sensor_check_1_dhw_charging"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_1_ERROR_RATE_PREPARING_DEFROST = (
        "refrigerant_temperature_sensor_check_1_preparing_defrost"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_1_ERROR_RATE_DEFROST = (
        "refrigerant_temperature_sensor_check_1_defrost"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_1_ERROR_RATE_COOLING = (
        "refrigerant_temperature_sensor_check_1_cooling"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_2_ERROR_RATE_HEATING = (
        "refrigerant_temperature_sensor_check_2_heating"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_2_ERROR_RATE_STANDBY = (
        "refrigerant_temperature_sensor_check_2_standby"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_2_ERROR_RATE_DHW_CHARGING = (
        "refrigerant_temperature_sensor_check_2_dhw_charging"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_2_ERROR_RATE_PREPARING_DEFROST = (
        "refrigerant_temperature_sensor_check_2_preparing_defrost"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_2_ERROR_RATE_DEFROST = (
        "refrigerant_temperature_sensor_check_2_defrost"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_2_ERROR_RATE_COOLING = (
        "refrigerant_temperature_sensor_check_2_cooling"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_3_ERROR_RATE_HEATING = (
        "refrigerant_temperature_sensor_check_3_heating"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_3_ERROR_RATE_STANDBY = (
        "refrigerant_temperature_sensor_check_3_standby"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_3_ERROR_RATE_DHW_CHARGING = (
        "refrigerant_temperature_sensor_check_3_dhw_charging"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_3_ERROR_RATE_PREPARING_DEFROST = (
        "refrigerant_temperature_sensor_check_3_preparing_defrost"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_3_ERROR_RATE_DEFROST = (
        "refrigerant_temperature_sensor_check_3_defrost"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_3_ERROR_RATE_COOLING = (
        "refrigerant_temperature_sensor_check_3_cooling"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_4_ERROR_RATE_HEATING = (
        "refrigerant_temperature_sensor_check_4_heating"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_4_ERROR_RATE_STANDBY = (
        "refrigerant_temperature_sensor_check_4_standby"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_4_ERROR_RATE_DHW_CHARGING = (
        "refrigerant_temperature_sensor_check_4_dhw_charging"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_4_ERROR_RATE_PREPARING_DEFROST = (
        "refrigerant_temperature_sensor_check_4_preparing_defrost"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_4_ERROR_RATE_DEFROST = (
        "refrigerant_temperature_sensor_check_4_defrost"
    )

    REFRIGERANT_TEMPERATURE_SENSOR_CHECK_4_ERROR_RATE_COOLING = (
        "refrigerant_temperature_sensor_check_4_cooling"
    )

    EVAPORATION_ABOVE_PRIMARY_INLET_TEMPERATURE_INTEGRAL = (
        "evaporation_above_primary_inlet_temperature_integral"
    )

    EVAPORATION_BELOW_PRIMARY_INLET_TEMPERATURE_INTEGRAL = (
        "evaporation_below_primary_inlet_temperature_integral"
    )

    POWER_CONSUMPTION_DHW = "power_consumption_dhw"

    POWER_CONSUMPTION_COOLING = "power_consumption_cooling"

    POWER_CONSUMPTION_HEATING = "power_consumption_heating"

    POWER_CONSUMPTION_TOTAL = "power_consumption_total"

    CUMULATIVE_POWER_CONSUMPTION_DHW = "cumulative_power_consumption_dhw"

    CUMULATIVE_POWER_CONSUMPTION_COOLING = "cumulative_power_consumption_cooling"

    CUMULATIVE_POWER_CONSUMPTION_HEATING = "cumulative_power_consumption_heating"

    CUMULATIVE_POWER_CONSUMPTION_TOTAL = "cumulative_power_consumption_total"

    HEAT_PRODUCTION_DHW = "heat_production_dhw"

    HEAT_PRODUCTION_COOLING = "heat_production_cooling"

    HEAT_PRODUCTION_HEATING = "heat_production_heating"

    HEAT_PRODUCTION_TOTAL = "heat_production_total"

    CUMULATIVE_HEAT_PRODUCTION_DHW = "cumulative_heat_production_dhw"

    CUMULATIVE_HEAT_PRODUCTION_COOLING = "cumulative_heat_production_cooling"

    CUMULATIVE_HEAT_PRODUCTION_HEATING = "cumulative_heat_production_heating"

    CUMULATIVE_HEAT_PRODUCTION_TOTAL = "cumulative_heat_production_total"

    HEATING_CURVE_SLOPE_HC0 = "heating_curve_slope_hc_0"

    HEATING_CURVE_SLOPE_HC1 = "heating_curve_slope_hc_1"

    HEATING_CURVE_SLOPE_HC2 = "heating_curve_slope_hc_2"

    HEATING_CURVE_SLOPE_HC3 = "heating_curve_slope_hc_3"

    HEATING_CURVE_SHIFT_HC0 = "heating_curve_shift_hc_0"

    HEATING_CURVE_SHIFT_HC1 = "heating_curve_shift_hc_1"

    HEATING_CURVE_SHIFT_HC2 = "heating_curve_shift_hc_2"

    HEATING_CURVE_SHIFT_HC3 = "heating_curve_shift_hc_3"

    OPERATING_PROGRAMS_NORMAL_TEMPERATURE_HC0 = (
        "operating_programs_normal_temperature_hc_0"
    )

    OPERATING_PROGRAMS_NORMAL_TEMPERATURE_HC1 = (
        "operating_programs_normal_temperature_hc_1"
    )

    OPERATING_PROGRAMS_NORMAL_TEMPERATURE_HC2 = (
        "operating_programs_normal_temperature_hc_2"
    )

    OPERATING_PROGRAMS_NORMAL_TEMPERATURE_HC3 = (
        "operating_programs_normal_temperature_hc_3"
    )

    OPERATING_PROGRAMS_COMFORT_TEMPERATURE_HC0 = (
        "operating_programs_comfort_temperature_hc_0"
    )

    OPERATING_PROGRAMS_COMFORT_TEMPERATURE_HC1 = (
        "operating_programs_comfort_temperature_hc_1"
    )

    OPERATING_PROGRAMS_COMFORT_TEMPERATURE_HC2 = (
        "operating_programs_comfort_temperature_hc_2"
    )

    OPERATING_PROGRAMS_COMFORT_TEMPERATURE_HC3 = (
        "operating_programs_comfort_temperature_hc_3"
    )

    OPERATING_PROGRAMS_REDUCED_TEMPERATURE_HC0 = (
        "operating_programs_reduced_temperature_hc_0"
    )

    OPERATING_PROGRAMS_REDUCED_TEMPERATURE_HC1 = (
        "operating_programs_reduced_temperature_hc_1"
    )

    OPERATING_PROGRAMS_REDUCED_TEMPERATURE_HC2 = (
        "operating_programs_reduced_temperature_hc_2"
    )

    OPERATING_PROGRAMS_REDUCED_TEMPERATURE_HC3 = (
        "operating_programs_reduced_temperature_hc_3"
    )

    DEFROST_PREPARATION_HOT_GAS_START_IQM = "defrost_preparation_hot_gas_start_iqm"

    DEFROST_PREPARATION_HOT_GAS_END_IQM = "defrost_preparation_hot_gas_end_iqm"

    DEFROST_PREPARATION_HOT_GAS_DIFF_IQM = "defrost_preparation_hot_gas_diff_iqm"

    DEFROST_HOT_GAS_START_IQM = "defrost_hot_gas_start_iqm"

    DEFROST_HOT_GAS_END_IQM = "defrost_hot_gas_end_iqm"

    DEFROST_HOT_GAS_DIFF_IQM = "defrost_hot_gas_diff_iqm"

    DEFROST_EVAPORATION_TEMP_START_IQM = "defrost_evaporation_temp_start_iqm"

    DEFROST_EVAPORATION_TEMP_END_IQM = "defrost_evaporation_temp_end_iqm"

    DEFROST_EVAPORATION_TEMP_DIFF_IQM = "defrost_evaporation_temp_diff_iqm"

    INLET_TO_OUTSIDE_TEMP_CORR = "inlet_to_outside_temp_corr"

    INLET_TO_OUTSIDE_TEMP_MAE = "inlet_to_outside_temp_mae"

    CONFIG_SCREED_DRYING_ACTIVE = "config_screed_drying_active"

    CONFIG_MODE_ON_ERROR = "config_mode_on_error"

    BLOCKING_HP_ERRORS = "blocking_hp_errors"

    EVU_LOCK_DURATION = "evu_lock_duration"

    HEATINGROD_POWER_CONSUMPTION_DHW = "heatingrod_power_consumption_dhw"

    HEATINGROD_POWER_CONSUMPTION_HEATING = "heatingrod_power_consumption_heating"

    HEATINGROD_POWER_CONSUMPTION_TOTAL = "heatingrod_power_consumption_total"

    HEATINGROD_CUMULATIVE_POWER_CONSUMPTION_DHW = (
        "heatingrod_cumulative_power_consumption_dhw"
    )

    HEATINGROD_CUMULATIVE_POWER_CONSUMPTION_HEATING = (
        "heatingrod_cumulative_power_consumption_heating"
    )

    HEATINGROD_CUMULATIVE_POWER_CONSUMPTION_TOTAL = (
        "heatingrod_cumulative_power_consumption_total"
    )

    SMART_ROOM_CONTROL_ACTIVE = "smart_room_control_active"

    E3_GAS_DEVICE = "e3_gas_device"

    VAPOR_DEW_POINT_DIFFERENCE_HEATING_IQM = "vapor_to_dew_point_difference_heating_iqm"

    VAPOR_DEW_POINT_DIFFERENCE_DHW_IQM = "vapor_to_dew_point_difference_dhw_iqm"

    VAPOR_DEW_POINT_DIFFERENCE_COOLING_IQM = "vapor_to_dew_point_difference_cooling_iqm"

    BURNER_STARTS_AT_LOW_VOLUMETRIC_FLOW = "burner_starts_at_low_volumetric_flow"

    BURNER_STARTS_AT_MEDIUM_VOLUMETRIC_FLOW = "burner_starts_at_medium_volumetric_flow"

    BURNER_STARTS_AT_HIGH_VOLUMETRIC_FLOW = "burner_starts_at_high_volumetric_flow"

    SECONDARYHEATGENERATOR_STARTS = "shg_starts"

    SECONDARYHEATGENERATOR_HOURS = "shg_hours"

    SECONDARYHEATGENERATOR_STARTS_RESULTING_IN_TEMPERATURE_INCREASE = (
        "shg_starts_with_temp_increase"
    )

    SECONDARYHEATGENERATOR_MEDIAN_GAP_BETWEEN_STARTS = "shg_median_gap_between_starts"

    HYBRID_HEAT_SOURCE_ALTERNATIONS = "hybrid_heat_source_alternations"

    HYBRID_HOURS_BETWEEN_FIRST_AND_LAST_START = (
        "hybrid_hours_between_first_and_last_start"
    )

    SECONDARYHEATGENERATOR_HEAT_REQUESTS = "shg_heat_requests"

    SECONDARYHEATGENERATOR_HEAT_REQUESTS_RESULTING_IN_TEMPERATURE_INCREASE = (
        "shg_heat_requests_with_temp_increase"
    )


def get_name(name: str) -> FeatureName:
    return getattr(FeatureName, name)


def extend_feature_name(FeatureName):
    existing_names = [(f.name, f.value) for f in FeatureName]
    new_names = []
    for data in CRITICAL_FUEL_CELL_ERRORS.values():
        new_names.append((data["feature_name"].upper(), data["feature_name"].lower()))
    for data in BATTERY_STATE_OF_CHARGE_HISTOGRAMS.values():
        new_names.append((data["feature_name"].upper(), data["feature_name"].lower()))

    return Enum("FeatureName", names=existing_names + new_names)


FeatureName = extend_feature_name(FeatureName)


@unique
class Unit(Enum):
    # time
    HOUR = "hour"

    MINUTE = "minute"

    SECOND = "second"

    # temperature
    DEGREE_CELSIUS = "degree_celsius"

    KELVIN = "kelvin"

    KELVIN_PER_MINUTE = "kelvin_per_minute"

    MINUTE_PER_KELVIN = "minute_per_kelvin"

    KELVIN_HOURS = "kelvin_hours"

    KELVIN_MINUTES = "kelvin_minutes"

    # pressure
    BAR = "bar"

    MILLI_BAR = "milli_bar"

    # resistance
    OHM = "ohm"

    # energy
    KILO_WATT_HOURS_OR_CUBIC_METERS = "kilo_watt_hours_or_cubic_meters"

    KILO_WATT_HOURS = "kilo_watt_hours"

    WATT_HOURS = "watt_hours"

    WATT = "watt"

    AMPERE_HOURS = "ampere_hours"

    # radiation
    WATT_PER_SQUARE_METER = "watt_per_square_meter"

    # flow
    LITER_PER_HOUR = "liter_per_hour"

    # speed
    KILOMETERS_PER_HOUR = "kilometers_per_hour"

    ROUNDS_PER_SECOND = "rounds_per_second"

    # length
    KILOMETER = "kilometer"

    MILLIMETER = "millimeter"

    # generic types
    CATEGORICAL = "categorical"

    BOOLEAN = "boolean"

    NUMBER = "number"

    CORRELATION = "correlation"

    PERCENTAGE = "percentage"

    PERCENTAGE_MINUTE = "percentage_minute"

    PERCENTAGE_MINUTE_PER_KELVIN = "percentage_minute_per_kelvin"

    PERCENTAGE_PER_BAR = "percentage_per_bar"

    DECIBEL_MILLIWATT = "decibel_milliwatt"

    STARTS_PER_HOUR = "starts_per_hour"

    RATIO = "ratio"

    CUBIC_METER_PER_HOUR = "cubic_meter_per_hour"