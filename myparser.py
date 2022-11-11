import configparser
from instance_generator import EnvConfig, InstanceConfig, GenConfig, RLConfig
import Utils


class Parser(object):
    def __init__(self, parser):
        configp = configparser.ConfigParser()
        configp.read("config.ini")

        parser.add_argument('--model', nargs="?", help="'VRPVCSD' or 'VRPSD'?", type=str, metavar="model_type",
                            default=["VRPVCSD"])
        parser.add_argument('--operation', nargs="+", help="'train' or 'test'?", type=str, metavar="operation")

        parser.add_argument('--c', nargs="+", help="number of customers", type=int, metavar="n_customers")
        parser.add_argument('--v', nargs="+", help="number of vehicles", type=int, metavar="n_vehicles")
        parser.add_argument('--q', nargs="+", help="capacity of vehicle", type=int, metavar="capacity")
        parser.add_argument('--dl', nargs="?", help="duration limit of vehicle", type=float, metavar="dl",
                            default=[0])
        parser.add_argument('--sv', nargs="?", help="stochastic variablity", type=int, metavar="stoch_type",
                            default=[0])
        parser.add_argument('--trials', nargs="?", help="number of trials to train", type=int, metavar="trials",
                            default=100000)

        parser.add_argument('--base_address', nargs="?", help="base_address", type=str, metavar="base_address",
                            default="Models/")
        parser.add_argument('--density', nargs="?", help="density class of instances", type=int,
                            metavar="density_class")
        parser.add_argument('--instance_class', nargs="?", help="instance class according to solomon", type=str,
                            metavar="instance_class", default="r")
        parser.add_argument('--instance_count', nargs="?", help="number of instances", type=int,
                            metavar="instance_count")
        parser.add_argument('--nb', nargs="?", help="neighbor customers", type=int, metavar="nb", default=15)

        parser.add_argument('--code', nargs="?", help="code", type=str, metavar="code",
                            default="")

        parser.add_argument('--obs', nargs="?", help="use obs (1 or 0)", type=int, metavar="use_obs",
                            default=1)

        args = parser.parse_args()

        # Instances
        self.n_customers = args.c[0] if args.c is not None else 0
        self.n_vehicles = args.v[0] if args.v is not None else 0
        self.capacity = args.q
        self.duration_limit = args.dl
        self.stoch_type = args.sv
        self.density_class = args.density
        self.instance_class = args.instance_class

        # General simulator
        self.model_type = args.model[0]
        self.operation = args.operation[0]
        self.trials = args.trials
        self.code = args.code
        self.base_address = args.base_address
        self.nb = args.nb
        self.instance_count = args.instance_count
        self.use_obs = args.obs

        self.env_config = configp["Environment"]
        self.rl_config = configp["RL"]

    def get_env_config(self):
        env_args = {"service_area": Utils.str_to_arr(self.env_config["service_area"]),
                    "model_type": self.model_type,
                    "m": self.n_vehicles,
                    "hm_slice": Utils.str_to_arr(self.env_config["hm_slice"])}
        env_config = EnvConfig(**env_args)
        return env_config

    def get_instance_config(self):
        instance_args = {"n": self.n_customers,
                         "m": self.n_vehicles,
                         "capacity": self.capacity,
                         "duration_limit": self.duration_limit[0],
                         "real_duration_limit": self.duration_limit,
                         "density_class": self.density_class,
                         "stoch_type": self.stoch_type,
                         "depot": Utils.str_to_arr(self.env_config["depot"])}
        return InstanceConfig(**instance_args)

    def get_general_config(self):
        general_config = {"model_type": self.model_type,
                          "operation": self.operation,
                          "trials": self.trials,
                          "code": self.code,
                          "base_address": self.base_address,
                          "nb": self.nb,
                          "instance_count": self.instance_count,
                          "report_every": int(self.rl_config["report_every"])}
        return GenConfig(**general_config)

    def get_rl_config(self):
        nn = {"nb": self.nb,
              "trials": self.trials,
              "use_obs": self.use_obs}
        for k, v in self.rl_config.items():
            if k == "lr_decay":
                nn[k] = Utils.str_to_arr(v)
            elif k in ["gama", "lr", "update_prob"]:
                nn[k] = float(v)
            else:
                nn[k] = int(v)
        return RLConfig(**nn)
