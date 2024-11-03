import gymnasium
import torch.cuda
from agents.super_agent import SuperAgent
from runner import Runner


def main(agent_count: int) -> None:
    torch.set_default_device('cuda')
    super_agent = SuperAgent(train_agent_count=agent_count)
    runners = [Runner(env=gymnasium.make("CartPole-v1", render_mode="human"), agent=agent)
               for agent in super_agent.agents]
    try:
        while True:
            for runner in runners:
                runner.step()
            super_agent.train()
    except KeyboardInterrupt:
        for runner in runners:
            runner.close()
        super_agent.save()


if __name__ == '__main__':
    main(agent_count=2)
