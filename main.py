import argparse
import random
import tensorflow as tf
import Utils
import instance_generator
import myparser
import vrp
import vrpsd_solver


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    parser = myparser.Parser(p)
    normalize_instances = True

    env_config = parser.get_env_config()
    if normalize_instances:
        env_config.service_area[0] /= Utils.Norms.COORD
        env_config.service_area[1] /= Utils.Norms.COORD

    instance_config = parser.get_instance_config()
    gen_config = parser.get_general_config()
    rl_config = parser.get_rl_config()

    # generate/load instances
    instances = None
    if parser.model_type == "VRPSD":
        instance_config.depot = [35, 35]
        env_config.service_area = [80, 80]
        instances = instance_generator.generate_vrpsd_instances(n_customers=parser.n_customers,
                                                                n_vehicles=parser.n_vehicles,
                                                                capacity=parser.capacity,
                                                                duration_limit=parser.duration_limit,
                                                                stoch_types=parser.stoch_type,
                                                                instance_config=instance_config,
                                                                normalize=normalize_instances)
    elif parser.model_type == "VRPVCSD":
        instance_config.depot = [.50, .50]
        env_config.service_area = [100, 100]
        if parser.operation in ["test"]:
            instances = instance_generator.load_VRPVCSD_instances(f"{parser.density_class}_{parser.capacity[0]}")
        else:
            instances = instance_generator.generate_VRPVCSD_instances_generalized(instance_config=instance_config,
                                                                                  density_class_list=[parser.density_class],
                                                                                  capacity_list=parser.capacity,
                                                                                  count=100)
            dls_list = set([i["Config"].duration_limit for i in instances])

    else:
        print("Model type is not defined.")

    # Initialize the environment
    vrpsd = vrp.VRPSD(env_config)

    if parser.operation == "train":
        caps = "".join([str(m) for m in instance_config.capacity])
        if parser.model_type == "VRPSD":
            dls = "".join([str(int(m)) for m in instance_config.real_duration_limit])
            svs = "".join([str(m) for m in instance_config.stoch_type])
            gen_config.code = f"{instance_config.n}_{instance_config.m}_{caps}_{dls}_{svs}_" \
                              f"{random.randint(1000, 9999)}"
        else:
            gen_config.code = f"{instance_config.density_class}_{caps}_" \
                              f"{random.randint(1000, 9999)}"

        print(f"Model code is {gen_config.code}")
        print("Params:")
        print("general config", gen_config)
        print("instance config", instance_config)
        print("rl config", rl_config)
        print("env config", env_config)

        with tf.Session() as sess:
            learner = vrpsd_solver.Learner(env=vrpsd, instances=instances, test_instance=instances[0],
                                           rl_config=rl_config, gen_config=gen_config, sess=sess)
            print("Trials\t#trains\treward\ttime\tloss")
            results = learner.train()
            # results = learner.train_centralized()
            learner.save_model()
        print("Done!")

    elif parser.operation == "test":
        with tf.Session() as sess:
            learner = vrpsd_solver.Learner(env=vrpsd, instances=instances, test_instance=instances[0],
                                           rl_config=rl_config, gen_config=gen_config, sess=sess)
            learner.load_model()
            rr = 0
            for e, instance in enumerate(instances[:200]):

                # if scenarios_set is not None:
                #     scenarios = np.array(scenarios_set[e]) / Utils.Norms.Q
                #     # scenarios = scenarios[:5]
                # else:
                #     scenarios = None
                # avg_rew = 0
                # for scenario in scenarios:
                #     res = vrp_sim.simulate(instance, scenario, method="random")
                #     avg_rew += res.final_reward
                # avg_rew /= len(scenarios)

                avg_rew = learner.test(instance, visualize=False)
                # print(avg_rew)
                if parser.model_type == "VRPVCSD":
                    print(f"{e + 1}\t{instance['Config'].real_n}\t"
                          # f"{instance['Config'].stoch_type}", 
                          f"{avg_rew}")
                else:
                    print(avg_rew)
    else:
        print("Operation is not defined.")

