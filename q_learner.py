class QLearner:
    
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1, epsilon_decay=1, td_lambda=0, batch_size=1, replay_size=1, copy_every=1, model=None, n_observations=None, n_actions=None):
        assert 0 <= alpha <= 1
        assert 0 <= gamma <= 1
        assert 0 <= epsilon <= 1
        assert 0 <= epsilon_decay
        assert 0 <= td_lambda < 1
        assert type(batch_size) == int
        assert batch_size > 0
        assert type(replay_size) == int
        assert replay_size > 0
        assert type(copy_every) == int
        assert (model is not None) or (n_observations is not None and n_actions is not None)
        if model == None:
            linear = pt.nn.Linear(n_observations, n_actions, bias=False)
            linear.weight.data.fill_(0)
            model = pt.nn.Sequential(
                linear,
            )
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.td_lambda = td_lambda
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.replay_memory = queue.deque(maxlen=self.replay_size)
        self.copy_every = copy_every
        self.n = 0
        self.k = 0
        self.model = model
        self.target_model = copy.deepcopy(self.model)
        self.criterion = pt.nn.MSELoss()
        self.optimizer = pt.optim.SGD(self.model.parameters(), lr=alpha)
        self.td_optimizer = pt.optim.SGD(self.model.parameters(), lr=alpha * (1 - td_lambda))
        
    def step(self, state, action, reward, next_state, next_action):
        self.n += 1
        if self.n % self.copy_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        q_values = self.model(pt.Tensor(state))
        q_value = q_values[np.argmax(action)]
        q_target = reward + self.gamma * self.target_model(pt.Tensor(next_state)).detach()[np.argmax(next_action)]
        q_targets = pt.Tensor([q_value for q_value in q_values])
        q_targets[np.argmax(action)] = q_target
        error = (q_target - q_value).item()
        if self.td_lambda == 0:
            self.td_optimizer.zero_grad()
        elif error == 0:
            self.replay_memory.append((state, action, reward, tuple(next_state), tuple(next_action)))
            return
        else:
            for parameter in self.model.parameters():
                try:
                    parameter.grad *= self.td_lambda * self.gamma
                except TypeError:
                    break
        q_value.backward()
        for parameter in self.model.parameters():
            parameter.grad *= -error
        self.td_optimizer.step()
        for parameter in self.model.parameters():
            parameter.grad /= -error
        self.replay_memory.append((state, action, reward, tuple(next_state), tuple(next_action)))
        
    def q_step(self, state, action, reward, next_state):
        self.n += 1
        if self.n % self.copy_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        q_values = self.model(pt.Tensor(state))
        next_q_values = self.target_model(pt.Tensor(next_state)).detach()
        q_target = reward + self.gamma * np.max(next_q_values)
        q_targets = pt.Tensor([q_value for q_value in q_values])
        q_targets[np.argmax(action)] = q_target
        loss = self.criterion(q_targets, q_values)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.replay_memory.append((state, action, reward, tuple(next_state), tuple(np.eye(len(self.n_actions))[np.argmax(next_q_values)])))
                
    def replay_step(self):
        batch = np.array(self.replay_memory)[np.random.randint(len(self.replay_memory), size=(self.batch_size))]
        q_values = self.model(pt.Tensor([np.array(state) for state in batch[:, 0]]))
        next_q_values = self.target_model(pt.Tensor([np.array(next_state) for next_state in batch[:, 3]])).detach()
        actions = np.argmax([np.array(action) for action in  batch[:, 1]], axis=1)
        rewards = np.array([np.array(reward) for reward in batch[:, 2]])
        next_actions = np.argmax([np.array(next_action) for next_action in  batch[:, 4]], axis=1)
        q_target = rewards + self.gamma * next_q_values.numpy()[(np.arange(len(next_q_values)), next_actions)]
        q_targets = pt.Tensor([list(q_value) for q_value in q_values])
        q_targets[(np.arange(len(next_q_values)), actions)] = pt.Tensor(q_target)
        loss = self.criterion(q_targets, q_values)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def reset_traces(self):
        self.td_optimizer.zero_grad()

    def predict_q_values(self, state):
        q_values = self.model(pt.Tensor(state))
        return q_values.detach().numpy()
    
    def predict_state_value(self, state):
        q_values = self.model(pt.Tensor(state))
        return np.max(q_values.detach())
    
    def greedy_action(self, state):
        self.k += 1
        q_values = self.model(pt.Tensor(state))
        return np.eye(self.n_actions)[np.argmax(q_values.detach())]
    
    def epsilon_greedy_action(self, state):
        self.k += 1
        if np.random.random() > self.epsilon / (1 + self.k * self.epsilon_decay):
            q_values = self.model(pt.Tensor(state))
            return np.eye(self.n_actions)[np.argmax(q_values.detach())]
        else:
            return np.eye(self.n_actions)[np.random.randint(self.n_actions)]
        
    def random_action(self, state):
        self.k += 1
        return np.eye(self.n_actions)[np.random.randint(self.n_actions)]
