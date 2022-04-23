### koopman embeding plus control

#### pendulum gym structure
ingredients.append(nn.Linear(input_s, 6))
ingredients.append(Activation())
ingredients.append(nn.Linear(6, 18))
ingredients.append(Activation())
ingredients.append(nn.Linear(18, 50))
ingredients.append(Activation())
ingredients.append(nn.Linear(50, 80))
ingredients.append(Activation())
ingredients.append(nn.Linear(80, output))

