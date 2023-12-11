from envs import ALFWorldEnv
from stable_baselines3 import A2C
from models import CustomActorCriticPolicy
from models import CustomExtractor

EPOCH = 1000

if __name__ == "__main__":

    env = ALFWorldEnv(100)
    print(f"task in this environment: {env.task}")
    
    model = A2C(
        CustomActorCriticPolicy,
        env,
        verbose=1,
        policy_kwargs={
            "features_extractor_class":CustomExtractor,
        }
    )

    for i in range(EPOCH):
        model.learn(
            total_timesteps=1000,
            reset_num_timesteps=False,
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
        )

    # inference
    myseed = 0
    done = False
    enc_obs, infos = env.reset(myseed)
    print(f"observation: {infos['obs'][0]}")
    while not done:
        action, _state = model.predict(enc_obs, deterministic=True)
        enc_obs, reward, done, _, infos = env.step(action)
        print(f"observation: {infos['obs'][0]}")