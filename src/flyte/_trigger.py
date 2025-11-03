from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal, Mapping, Union

import rich.repr

Timezone = Literal[
    "Etc/GMT-5",
    "Europe/London",
    "America/Martinique",
    "Asia/Sakhalin",
    "Europe/Podgorica",
    "America/Grand_Turk",
    "America/Dawson_Creek",
    "Africa/Asmera",
    "Canada/Newfoundland",
    "CST6CDT",
    "ROC",
    "Asia/Taipei",
    "America/Danmarkshavn",
    "Asia/Yakutsk",
    "America/Catamarca",
    "Asia/Gaza",
    "America/Chihuahua",
    "Africa/Lusaka",
    "Atlantic/Azores",
    "Canada/Atlantic",
    "Africa/Abidjan",
    "America/Port-au-Prince",
    "Pacific/Noumea",
    "Asia/Barnaul",
    "Europe/Bucharest",
    "Asia/Samarkand",
    "Africa/Bangui",
    "America/Yakutat",
    "Africa/Porto-Novo",
    "Etc/GMT+12",
    "America/Fortaleza",
    "Australia/Brisbane",
    "America/Goose_Bay",
    "America/Nassau",
    "Arctic/Longyearbyen",
    "Asia/Kolkata",
    "Indian/Mayotte",
    "Asia/Tel_Aviv",
    "America/Cambridge_Bay",
    "Africa/Nouakchott",
    "Australia/North",
    "Europe/Tirane",
    "America/Ensenada",
    "Asia/Rangoon",
    "Pacific/Kanton",
    "Africa/Tunis",
    "Europe/Kyiv",
    "America/Halifax",
    "Europe/Guernsey",
    "America/Cancun",
    "Canada/Saskatchewan",
    "Europe/Helsinki",
    "Pacific/Norfolk",
    "Chile/Continental",
    "Eire",
    "Africa/Mogadishu",
    "Pacific/Midway",
    "Etc/GMT-1",
    "Etc/UTC",
    "America/Argentina/San_Luis",
    "Europe/Gibraltar",
    "GMT+0",
    "HST",
    "America/Kentucky/Louisville",
    "US/Alaska",
    "Africa/Addis_Ababa",
    "America/Costa_Rica",
    "Pacific/Rarotonga",
    "America/Matamoros",
    "Europe/Vienna",
    "America/Vancouver",
    "Mexico/BajaNorte",
    "America/Merida",
    "Pacific/Efate",
    "America/La_Paz",
    "Pacific/Marquesas",
    "America/Manaus",
    "Antarctica/South_Pole",
    "Pacific/Easter",
    "Europe/Mariehamn",
    "Atlantic/Madeira",
    "Pacific/Bougainville",
    "Antarctica/Syowa",
    "America/Montevideo",
    "Africa/Khartoum",
    "America/St_Thomas",
    "Africa/Mbabane",
    "America/Campo_Grande",
    "Australia/Queensland",
    "Asia/Damascus",
    "Etc/Universal",
    "Asia/Amman",
    "Europe/Samara",
    "Australia/Tasmania",
    "Africa/Douala",
    "Indian/Antananarivo",
    "Canada/Eastern",
    "America/Argentina/Rio_Gallegos",
    "Pacific/Samoa",
    "Australia/Canberra",
    "Australia/Sydney",
    "Atlantic/Faeroe",
    "Asia/Ust-Nera",
    "Pacific/Palau",
    "Africa/Kinshasa",
    "Asia/Makassar",
    "Asia/Hebron",
    "ROK",
    "America/St_Vincent",
    "America/Argentina/Buenos_Aires",
    "Africa/Asmara",
    "Indian/Mahe",
    "America/Dawson",
    "Europe/Lisbon",
    "Pacific/Pago_Pago",
    "Brazil/West",
    "America/Santarem",
    "America/Virgin",
    "America/Indianapolis",
    "Asia/Bangkok",
    "America/Indiana/Marengo",
    "Atlantic/St_Helena",
    "Europe/Moscow",
    "Europe/Istanbul",
    "Asia/Famagusta",
    "Asia/Chongqing",
    "America/Cuiaba",
    "America/Detroit",
    "America/Swift_Current",
    "Greenwich",
    "Antarctica/Davis",
    "Africa/Conakry",
    "America/Asuncion",
    "Asia/Hovd",
    "Africa/Freetown",
    "America/Bahia_Banderas",
    "Etc/GMT+6",
    "Indian/Chagos",
    "Pacific/Kiritimati",
    "America/Toronto",
    "EET",
    "Indian/Cocos",
    "America/Caracas",
    "America/Indiana/Knox",
    "America/Indiana/Winamac",
    "Asia/Katmandu",
    "America/Dominica",
    "Asia/Hong_Kong",
    "GMT0",
    "Atlantic/Faroe",
    "US/Michigan",
    "UCT",
    "Etc/GMT-9",
    "America/Cordoba",
    "Etc/GMT",
    "Chile/EasterIsland",
    "America/Resolute",
    "America/Juneau",
    "Europe/Chisinau",
    "Africa/Djibouti",
    "Antarctica/Vostok",
    "Europe/Ulyanovsk",
    "US/Hawaii",
    "Africa/Juba",
    "America/Chicago",
    "America/Boa_Vista",
    "Antarctica/DumontDUrville",
    "Brazil/East",
    "Mexico/BajaSur",
    "Africa/Lubumbashi",
    "America/Anguilla",
    "Etc/GMT-13",
    "Canada/Central",
    "Europe/Busingen",
    "America/Ciudad_Juarez",
    "America/Edmonton",
    "Atlantic/South_Georgia",
    "America/Anchorage",
    "America/Rosario",
    "America/Araguaina",
    "Asia/Shanghai",
    "America/Tijuana",
    "America/Cayenne",
    "America/Regina",
    "Australia/NSW",
    "America/Santa_Isabel",
    "Indian/Comoro",
    "Europe/San_Marino",
    "WET",
    "Poland",
    "US/Indiana-Starke",
    "Asia/Saigon",
    "Africa/Ceuta",
    "Pacific/Niue",
    "Australia/Darwin",
    "Asia/Yekaterinburg",
    "Pacific/Chuuk",
    "Asia/Kathmandu",
    "Asia/Almaty",
    "America/Miquelon",
    "Asia/Choibalsan",
    "Australia/Melbourne",
    "America/Managua",
    "Portugal",
    "Iceland",
    "Africa/Malabo",
    "America/North_Dakota/Center",
    "Asia/Macau",
    "EST5EDT",
    "America/Louisville",
    "America/Fort_Nelson",
    "Europe/Amsterdam",
    "America/Rio_Branco",
    "Etc/GMT0",
    "America/Thule",
    "Pacific/Fakaofo",
    "Africa/Bujumbura",
    "Asia/Baku",
    "Etc/GMT+8",
    "Etc/GMT+10",
    "America/Argentina/Tucuman",
    "Africa/Lagos",
    "Europe/Paris",
    "Indian/Christmas",
    "Asia/Qatar",
    "Australia/ACT",
    "Asia/Pyongyang",
    "America/North_Dakota/Beulah",
    "Factory",
    "Europe/Copenhagen",
    "Pacific/Fiji",
    "America/Lower_Princes",
    "Antarctica/Macquarie",
    "America/Punta_Arenas",
    "Antarctica/Rothera",
    "America/Montserrat",
    "Etc/GMT+0",
    "NZ",
    "America/Argentina/La_Rioja",
    "America/Argentina/Catamarca",
    "Antarctica/Troll",
    "Europe/Vatican",
    "Cuba",
    "Africa/Windhoek",
    "America/Havana",
    "Asia/Atyrau",
    "Australia/Eucla",
    "America/Guatemala",
    "US/Mountain",
    "Europe/Saratov",
    "Asia/Jakarta",
    "US/Aleutian",
    "Asia/Thimphu",
    "Pacific/Enderbury",
    "Africa/Luanda",
    "Europe/Kirov",
    "Pacific/Tongatapu",
    "Africa/Sao_Tome",
    "Africa/El_Aaiun",
    "Iran",
    "America/Godthab",
    "MST7MDT",
    "Europe/Sofia",
    "America/Thunder_Bay",
    "Australia/South",
    "America/Tegucigalpa",
    "Africa/Monrovia",
    "Egypt",
    "America/Scoresbysund",
    "Singapore",
    "Etc/GMT-7",
    "Africa/Lome",
    "America/Metlakatla",
    "Asia/Singapore",
    "Pacific/Chatham",
    "America/Paramaribo",
    "Asia/Ulaanbaatar",
    "Antarctica/Mawson",
    "Australia/Yancowinna",
    "GMT-0",
    "America/Belize",
    "Asia/Magadan",
    "America/Fort_Wayne",
    "Pacific/Nauru",
    "Europe/Warsaw",
    "Asia/Muscat",
    "Europe/Sarajevo",
    "Etc/GMT-2",
    "Africa/Gaborone",
    "Africa/Bamako",
    "Asia/Qyzylorda",
    "GB",
    "Etc/GMT+2",
    "America/Lima",
    "Asia/Omsk",
    "Kwajalein",
    "Asia/Tokyo",
    "America/Inuvik",
    "Asia/Tashkent",
    "Jamaica",
    "America/Argentina/ComodRivadavia",
    "Africa/Algiers",
    "Africa/Harare",
    "America/Argentina/San_Juan",
    "Pacific/Guam",
    "Europe/Astrakhan",
    "Africa/Nairobi",
    "US/Arizona",
    "EST",
    "Australia/Hobart",
    "America/Kentucky/Monticello",
    "Asia/Urumqi",
    "Etc/GMT-14",
    "Pacific/Johnston",
    "Etc/Zulu",
    "Asia/Ashgabat",
    "Atlantic/Jan_Mayen",
    "America/Aruba",
    "America/Argentina/Jujuy",
    "Etc/Greenwich",
    "America/St_Lucia",
    "Australia/Currie",
    "Asia/Jerusalem",
    "America/Atka",
    "America/St_Kitts",
    "America/El_Salvador",
    "Europe/Riga",
    "US/Central",
    "Etc/GMT+3",
    "America/Montreal",
    "Australia/Lord_Howe",
    "W-SU",
    "America/Noronha",
    "Canada/Mountain",
    "America/Indiana/Vincennes",
    "Europe/Simferopol",
    "Pacific/Gambier",
    "Africa/Tripoli",
    "Asia/Novosibirsk",
    "Navajo",
    "Asia/Harbin",
    "America/Rankin_Inlet",
    "Asia/Kuching",
    "America/Argentina/Salta",
    "Europe/Bratislava",
    "America/Glace_Bay",
    "America/Argentina/Mendoza",
    "Asia/Tomsk",
    "America/Nipigon",
    "Asia/Pontianak",
    "Australia/Perth",
    "Indian/Reunion",
    "Europe/Uzhgorod",
    "Europe/Athens",
    "Brazil/DeNoronha",
    "Zulu",
    "Asia/Qostanay",
    "Europe/Malta",
    "Indian/Maldives",
    "Asia/Jayapura",
    "America/Denver",
    "Atlantic/Reykjavik",
    "Australia/West",
    "America/Phoenix",
    "Europe/Volgograd",
    "Asia/Kamchatka",
    "America/Kralendijk",
    "America/Creston",
    "Africa/Dakar",
    "Europe/Andorra",
    "Europe/Madrid",
    "Japan",
    "Pacific/Kosrae",
    "GMT",
    "America/Maceio",
    "America/Porto_Acre",
    "Asia/Ho_Chi_Minh",
    "Asia/Kashgar",
    "US/Samoa",
    "Africa/Banjul",
    "Asia/Anadyr",
    "Etc/GMT-6",
    "Pacific/Truk",
    "America/Winnipeg",
    "Africa/Ndjamena",
    "Africa/Bissau",
    "Asia/Baghdad",
    "Israel",
    "America/Guadeloupe",
    "America/Buenos_Aires",
    "America/Adak",
    "Asia/Vladivostok",
    "Pacific/Tarawa",
    "Antarctica/Casey",
    "Antarctica/Palmer",
    "Asia/Irkutsk",
    "Asia/Colombo",
    "America/Port_of_Spain",
    "America/North_Dakota/New_Salem",
    "Europe/Dublin",
    "Pacific/Ponape",
    "America/Boise",
    "Pacific/Yap",
    "America/Whitehorse",
    "PRC",
    "Australia/Adelaide",
    "America/Indiana/Vevay",
    "Europe/Berlin",
    "America/Recife",
    "Europe/Oslo",
    "Turkey",
    "Europe/Luxembourg",
    "Europe/Zagreb",
    "America/Grenada",
    "Africa/Blantyre",
    "Asia/Tbilisi",
    "America/Coyhaique",
    "Pacific/Apia",
    "Africa/Niamey",
    "America/Guyana",
    "Asia/Yerevan",
    "Pacific/Honolulu",
    "America/Hermosillo",
    "Asia/Macao",
    "Europe/Belfast",
    "America/Indiana/Tell_City",
    "Asia/Dushanbe",
    "Asia/Novokuznetsk",
    "Africa/Maseru",
    "Pacific/Funafuti",
    "Antarctica/McMurdo",
    "America/Menominee",
    "NZ-CHAT",
    "MET",
    "Asia/Dhaka",
    "America/Jujuy",
    "Europe/Vaduz",
    "Europe/Budapest",
    "Asia/Kuwait",
    "Africa/Maputo",
    "Asia/Aqtau",
    "Europe/Belgrade",
    "Africa/Ouagadougou",
    "America/Puerto_Rico",
    "Europe/Vilnius",
    "Asia/Chita",
    "America/Yellowknife",
    "America/Ojinaga",
    "America/Shiprock",
    "America/Bahia",
    "America/Tortola",
    "America/Antigua",
    "Etc/GMT+11",
    "Atlantic/Bermuda",
    "Asia/Khandyga",
    "US/Pacific",
    "Asia/Nicosia",
    "Etc/GMT-3",
    "Asia/Kabul",
    "America/St_Johns",
    "Etc/GMT+5",
    "Asia/Dubai",
    "Pacific/Galapagos",
    "Etc/GMT-0",
    "America/Indiana/Petersburg",
    "America/Blanc-Sablon",
    "Etc/GMT-10",
    "Pacific/Tahiti",
    "America/Argentina/Cordoba",
    "Europe/Tiraspol",
    "America/Pangnirtung",
    "Africa/Casablanca",
    "Brazil/Acre",
    "Pacific/Pitcairn",
    "Europe/Ljubljana",
    "Africa/Cairo",
    "America/Nuuk",
    "Asia/Chungking",
    "Africa/Dar_es_Salaam",
    "Asia/Bahrain",
    "Etc/GMT-12",
    "Asia/Krasnoyarsk",
    "US/East-Indiana",
    "Europe/Minsk",
    "Asia/Dili",
    "Etc/GMT+7",
    "Asia/Seoul",
    "Asia/Yangon",
    "Europe/Zurich",
    "America/Knox_IN",
    "Atlantic/Cape_Verde",
    "Asia/Ashkhabad",
    "Pacific/Port_Moresby",
    "Europe/Prague",
    "Africa/Kampala",
    "America/Belem",
    "Asia/Vientiane",
    "America/Moncton",
    "Europe/Monaco",
    "America/Mazatlan",
    "Africa/Brazzaville",
    "America/Marigot",
    "Asia/Dacca",
    "Etc/GMT-11",
    "Atlantic/Stanley",
    "Asia/Kuala_Lumpur",
    "Hongkong",
    "Asia/Manila",
    "America/Santiago",
    "Indian/Mauritius",
    "Europe/Brussels",
    "Europe/Isle_of_Man",
    "Australia/Broken_Hill",
    "Asia/Brunei",
    "Europe/Zaporozhye",
    "America/New_York",
    "Asia/Tehran",
    "America/Panama",
    "Africa/Libreville",
    "America/Santo_Domingo",
    "Pacific/Wake",
    "Pacific/Auckland",
    "America/Argentina/Ushuaia",
    "Asia/Aden",
    "Africa/Timbuktu",
    "Asia/Bishkek",
    "Etc/GMT+1",
    "Pacific/Pohnpei",
    "America/Sao_Paulo",
    "US/Eastern",
    "Europe/Nicosia",
    "Pacific/Kwajalein",
    "America/Iqaluit",
    "America/Cayman",
    "America/Monterrey",
    "America/Guayaquil",
    "America/Nome",
    "America/Barbados",
    "Pacific/Majuro",
    "Etc/GMT-8",
    "Asia/Phnom_Penh",
    "America/Rainy_River",
    "America/Indiana/Indianapolis",
    "America/Atikokan",
    "America/St_Barthelemy",
    "PST8PDT",
    "Universal",
    "Indian/Kerguelen",
    "Etc/UCT",
    "Australia/LHI",
    "Europe/Stockholm",
    "Asia/Ujung_Pandang",
    "America/Los_Angeles",
    "Asia/Riyadh",
    "America/Curacao",
    "Africa/Johannesburg",
    "Etc/GMT+4",
    "Canada/Yukon",
    "GB-Eire",
    "Asia/Karachi",
    "America/Mendoza",
    "Australia/Victoria",
    "America/Jamaica",
    "Australia/Lindeman",
    "Asia/Srednekolymsk",
    "America/Bogota",
    "Asia/Beirut",
    "Asia/Calcutta",
    "MST",
    "Europe/Jersey",
    "Etc/GMT+9",
    "Asia/Ulan_Bator",
    "Europe/Rome",
    "Pacific/Wallis",
    "Etc/GMT-4",
    "Libya",
    "UTC",
    "Asia/Thimbu",
    "Canada/Pacific",
    "Africa/Kigali",
    "America/Eirunepe",
    "Europe/Kaliningrad",
    "Atlantic/Canary",
    "America/Mexico_City",
    "Europe/Kiev",
    "CET",
    "Europe/Skopje",
    "Pacific/Saipan",
    "America/Sitka",
    "Africa/Accra",
    "Asia/Aqtobe",
    "Asia/Oral",
    "Pacific/Guadalcanal",
    "Asia/Istanbul",
    "America/Porto_Velho",
    "America/Coral_Harbour",
    "Mexico/General",
    "Europe/Tallinn",
]


class _trigger_time:
    """
    This class represents the actual time of Trigger, which can be bound to any task input.
    """


TriggerTime = _trigger_time()


@rich.repr.auto
@dataclass(frozen=True)
class Cron:
    """
    This class defines a Cron automation that can be associated with a Trigger in Flyte.
    Example usage:
    ```python
    from flyte.trigger import Trigger, Cron
    my_trigger = Trigger(
        name="my_cron_trigger",
        automation=Cron("0 * * * *"),  # Runs every hour
        description="A trigger that runs every hour",
    )
    ```
    """

    expression: str
    timezone: Timezone = "UTC"

    @property
    def timezone_expression(self) -> str:
        return f"CRON_TZ={self.timezone} {self.expression}"

    def __str__(self):
        return f"Cron Trigger: {self.timezone_expression}"


@rich.repr.auto
@dataclass(frozen=True)
class FixedRate:
    """
    This class defines a FixedRate automation that can be associated with a Trigger in Flyte.
    Example usage:
    ```python
    from flyte.trigger import Trigger, FixedRate
    my_trigger = Trigger(
        name="my_fixed_rate_trigger",
        automation=FixedRate(60),  # Runs every hour
        description="A trigger that runs every hour",
    )
    ```
    """

    interval_minutes: int
    start_time: datetime | None = None

    def __str__(self):
        return f"FixedRate Trigger: every {self.interval_minutes} minutes"


@rich.repr.auto
@dataclass(frozen=True)
class Trigger:
    """
    This class defines specification of a Trigger, that can be associated with any Flyte V2 task.
    The trigger then is deployed to the Flyte Platform.

    Triggers can be used to run tasks on a schedule, in response to events, or based on other conditions.
    The `Trigger` class encapsulates the metadata and configuration needed to define a trigger.

    You can associate the same Trigger object with multiple tasks.

    Example usage:
    ```python
    from flyte.trigger import Trigger
    my_trigger = Trigger(
        name="my_trigger",
        description="A trigger that runs every hour",
    )
    ```

    :param name: (str) The name of the trigger.
    :param automation: (AutomationType) The automation type, currently only supports Cron.
    :param description: (str) A description of the trigger, default is an empty string.
    :param auto_activate: (bool) Whether the trigger should be automatically activated, default is True.
    :param inputs: (Dict[str, Any]) Optional inputs for the trigger, default is None. If provided, will replace the
       values for inputs to these defaults.
    :param env_vars: (Dict[str, str]) Optional environment variables for the trigger, default is None. If provided, will
        replace the environment variables set in the config of the task.
    :param interruptible: (bool) Whether the trigger run is interruptible,
      default is None (maintains the configured behavior). If provided, it overrides whatever is set in the config
      of the task.
    :param overwrite_cache: (bool) Whether to overwrite the cache, default is False.
    :param queue: (str) Optional queue to run the trigger in, default is None.
    :param labels: (Mapping[str, str]) Optional labels to attach to the trigger, default is None.
    :param annotations: (Mapping[str, str]) Optional annotations to attach to the trigger, default is None.
    """

    name: str
    automation: Union[Cron, FixedRate]
    description: str = ""
    auto_activate: bool = True
    inputs: Dict[str, Any] | None = None
    env_vars: Dict[str, str] | None = None
    interruptible: bool | None = None
    overwrite_cache: bool = False
    queue: str | None = None
    labels: Mapping[str, str] | None = None
    annotations: Mapping[str, str] | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Trigger name cannot be empty")
        if self.automation is None:
            raise ValueError("Automation cannot be None")

    @classmethod
    def daily(
        cls,
        trigger_time_input_key: str = "trigger_time",
        *,
        name: str = "daily",
        description: str = "A trigger that runs daily at midnight",
        auto_activate: bool = True,
        inputs: Dict[str, Any] | None = None,
        env_vars: Dict[str, str] | None = None,
        interruptible: bool | None = None,
        overwrite_cache: bool = False,
        queue: str | None = None,
        labels: Mapping[str, str] | None = None,
        annotations: Mapping[str, str] | None = None,
    ) -> Trigger:
        """
        Creates a Cron trigger that runs daily at midnight.

        Args:
            trigger_time_input_key (str): The input key for the trigger time, default is "trigger_time".
            name (str): The name of the trigger, default is "daily".
            description (str): A description of the trigger.
            auto_activate (bool): Whether the trigger should be automatically activated.
            inputs (Dict[str, Any] | None): Optional inputs for the trigger.
            env_vars (Dict[str, str] | None): Optional environment variables.
            interruptible (bool | None): Whether the triggered run is interruptible.
            overwrite_cache (bool): Whether to overwrite the cache.
            queue (str | None): Optional queue to run the trigger in.
            labels (Mapping[str, str] | None): Optional labels to attach to the trigger.
            annotations (Mapping[str, str] | None): Optional annotations to attach to the trigger.

        Returns:
            Trigger: A trigger that runs daily at midnight.
        """
        final_inputs = {trigger_time_input_key: TriggerTime}
        if inputs:
            final_inputs.update(inputs)

        return cls(
            name=name,
            automation=Cron("0 0 * * *"),  # Cron expression for daily at midnight
            description=description,
            auto_activate=auto_activate,
            inputs=final_inputs,
            env_vars=env_vars,
            interruptible=interruptible,
            overwrite_cache=overwrite_cache,
            queue=queue,
            labels=labels,
            annotations=annotations,
        )

    @classmethod
    def hourly(
        cls,
        trigger_time_input_key: str = "trigger_time",
        *,
        name: str = "hourly",
        description: str = "A trigger that runs every hour",
        auto_activate: bool = True,
        inputs: Dict[str, Any] | None = None,
        env_vars: Dict[str, str] | None = None,
        interruptible: bool | None = None,
        overwrite_cache: bool = False,
        queue: str | None = None,
        labels: Mapping[str, str] | None = None,
        annotations: Mapping[str, str] | None = None,
    ) -> Trigger:
        """
        Creates a Cron trigger that runs every hour.

        Args:
            trigger_time_input_key (str): The input parameter for the trigger time, default is "trigger_time".
            name (str): The name of the trigger, default is "hourly".
            description (str): A description of the trigger.
            auto_activate (bool): Whether the trigger should be automatically activated.
            inputs (Dict[str, Any] | None): Optional inputs for the trigger.
            env_vars (Dict[str, str] | None): Optional environment variables.
            interruptible (bool | None): Whether the trigger is interruptible.
            overwrite_cache (bool): Whether to overwrite the cache.
            queue (str | None): Optional queue to run the trigger in.
            labels (Mapping[str, str] | None): Optional labels to attach to the trigger.
            annotations (Mapping[str, str] | None): Optional annotations to attach to the trigger.

        Returns:
            Trigger: A trigger that runs every hour, on the hour.
        """
        final_inputs = {trigger_time_input_key: TriggerTime}
        if inputs:
            final_inputs.update(inputs)

        return cls(
            name=name,
            automation=Cron("0 * * * *"),  # Cron expression for every hour
            description=description,
            auto_activate=auto_activate,
            inputs=final_inputs,
            env_vars=env_vars,
            interruptible=interruptible,
            overwrite_cache=overwrite_cache,
            queue=queue,
            labels=labels,
            annotations=annotations,
        )

    @classmethod
    def minutely(
        cls,
        trigger_time_input_key: str = "trigger_time",
        *,
        name: str = "minutely",
        description: str = "A trigger that runs every minute",
        auto_activate: bool = True,
        inputs: Dict[str, Any] | None = None,
        env_vars: Dict[str, str] | None = None,
        interruptible: bool | None = None,
        overwrite_cache: bool = False,
        queue: str | None = None,
        labels: Mapping[str, str] | None = None,
        annotations: Mapping[str, str] | None = None,
    ) -> Trigger:
        """
        Creates a Cron trigger that runs every minute.

        Args:
            trigger_time_input_key (str): The input parameter for the trigger time, default is "trigger_time".
            name (str): The name of the trigger, default is "every_minute".
            description (str): A description of the trigger.
            auto_activate (bool): Whether the trigger should be automatically activated.
            inputs (Dict[str, Any] | None): Optional inputs for the trigger.
            env_vars (Dict[str, str] | None): Optional environment variables.
            interruptible (bool | None): Whether the trigger is interruptible.
            overwrite_cache (bool): Whether to overwrite the cache.
            queue (str | None): Optional queue to run the trigger in.
            labels (Mapping[str, str] | None): Optional labels to attach to the trigger.
            annotations (Mapping[str, str] | None): Optional annotations to attach to the trigger.

        Returns:
            Trigger: A trigger that runs every minute.
        """
        final_inputs = {trigger_time_input_key: TriggerTime}
        if inputs:
            final_inputs.update(inputs)

        return cls(
            name=name,
            automation=Cron("* * * * *"),  # Cron expression for every minute
            description=description,
            auto_activate=auto_activate,
            inputs=final_inputs,
            env_vars=env_vars,
            interruptible=interruptible,
            overwrite_cache=overwrite_cache,
            queue=queue,
            labels=labels,
            annotations=annotations,
        )

    @classmethod
    def weekly(
        cls,
        trigger_time_input_key: str = "trigger_time",
        *,
        name: str = "weekly",
        description: str = "A trigger that runs weekly on Sundays at midnight",
        auto_activate: bool = True,
        inputs: Dict[str, Any] | None = None,
        env_vars: Dict[str, str] | None = None,
        interruptible: bool | None = None,
        overwrite_cache: bool = False,
        queue: str | None = None,
        labels: Mapping[str, str] | None = None,
        annotations: Mapping[str, str] | None = None,
    ) -> Trigger:
        """
        Creates a Cron trigger that runs weekly on Sundays at midnight.

        Args:
            trigger_time_input_key (str): The input parameter for the trigger time, default is "trigger_time".
            name (str): The name of the trigger, default is "weekly".
            description (str): A description of the trigger.
            auto_activate (bool): Whether the trigger should be automatically activated.
            inputs (Dict[str, Any] | None): Optional inputs for the trigger.
            env_vars (Dict[str, str] | None): Optional environment variables.
            interruptible (bool | None): Whether the trigger is interruptible.
            overwrite_cache (bool): Whether to overwrite the cache.
            queue (str | None): Optional queue to run the trigger in.
            labels (Mapping[str, str] | None): Optional labels to attach to the trigger.
            annotations (Mapping[str, str] | None): Optional annotations to attach to the trigger.

        Returns:
            Trigger: A trigger that runs weekly on Sundays at midnight.
        """
        final_inputs = {trigger_time_input_key: TriggerTime}
        if inputs:
            final_inputs.update(inputs)

        return cls(
            name=name,
            automation=Cron("0 0 * * 0"),  # Cron expression for every Sunday at midnight
            description=description,
            auto_activate=auto_activate,
            inputs=final_inputs,
            env_vars=env_vars,
            interruptible=interruptible,
            overwrite_cache=overwrite_cache,
            queue=queue,
            labels=labels,
            annotations=annotations,
        )

    @classmethod
    def monthly(
        cls,
        trigger_time_input_key: str = "trigger_time",
        *,
        name: str = "monthly",
        description: str = "A trigger that runs monthly on the 1st at midnight",
        auto_activate: bool = True,
        inputs: Dict[str, Any] | None = None,
        env_vars: Dict[str, str] | None = None,
        interruptible: bool | None = None,
        overwrite_cache: bool = False,
        queue: str | None = None,
        labels: Mapping[str, str] | None = None,
        annotations: Mapping[str, str] | None = None,
    ) -> Trigger:
        """
        Creates a Cron trigger that runs monthly on the 1st at midnight.

        Args:
            trigger_time_input_key (str): The input parameter for the trigger time, default is "trigger_time".
            name (str): The name of the trigger, default is "monthly".
            description (str): A description of the trigger.
            auto_activate (bool): Whether the trigger should be automatically activated.
            inputs (Dict[str, Any] | None): Optional inputs for the trigger.
            env_vars (Dict[str, str] | None): Optional environment variables.
            interruptible (bool | None): Whether the trigger is interruptible.
            overwrite_cache (bool): Whether to overwrite the cache.
            queue (str | None): Optional queue to run the trigger in.
            labels (Mapping[str, str] | None): Optional labels to attach to the trigger.
            annotations (Mapping[str, str] | None): Optional annotations to attach to the trigger.

        Returns:
            Trigger: A trigger that runs monthly on the 1st at midnight.
        """
        final_inputs = {trigger_time_input_key: TriggerTime}
        if inputs:
            final_inputs.update(inputs)

        return cls(
            name=name,
            automation=Cron("0 0 1 * *"),  # Cron expression for monthly on the 1st at midnight
            description=description,
            auto_activate=auto_activate,
            inputs=final_inputs,
            env_vars=env_vars,
            interruptible=interruptible,
            overwrite_cache=overwrite_cache,
            queue=queue,
            labels=labels,
            annotations=annotations,
        )


if __name__ == "__main__":
    from typing import get_args

    vals = get_args(Timezone)
    with open("/tmp/timezones.txt", "w") as f:
        for v in vals:
            c = Cron(expression="0 0 * * *", timezone=v)
            f.write(f"{c.timezone_expression}\n")
    print(f"Wrote {len(vals)} timezones to /tmp/timezones.txt")
