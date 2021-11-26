import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from helper import plot
import pickle

# Hiperparámetros
NUM_EPISODES = 10_000_000
LR = 0
DISCOUNT_RATE = 0.99
MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.3
EXPLORATION_DECAY_RATE = 0.1


class Agent:
    """
    Esta clase posee al agente y define sus comportamientos.
    """

    def __init__(self):
        """
        Este método inicializa los atributos del agente.
        """

        # TODO: Actividad 2.1: Inicializa la Q-Table.

        # Inicializamos los juegos realizados por el agente en 0.
        self.n_games = 0

    def get_state(self, game):
        """
        Este método consulta al juego por el estado del agente
        y lo retorna como una tupla. NO TOCAR.
        """

        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return tuple(state)

    def get_action(self, state):
        """
        Este método recibe una estado del agente y retorna un
        entero que representa a la acción.
        """
        # TODO: Actividad 2.2: Modifica este método para explorar o explotar
        return random.randint(0, 2)


def train():
    """
    Esta función es la encargada de entrenar al agente.
    """

    # Las siguientes variables nos permitirán llevar registro del
    # desempeño del agente.
    plot_scores = []
    plot_mean_scores = []
    mean_score = 0
    total_score = 0
    record = 0
    period_steps = 0
    period_score = 0

    # Instanciamos al agente o lo cargamos desde un pickle.
    agent = Agent()
    # agent = pickle.load(open("model/agent.p", "rb"))
    # Instanciamos el juego. El bool 'vis' define si queremos visualizar el juego o no.
    # Visualizarlo lo hace mucho más lento.
    game = SnakeGameAI(vis=False)
    # Inicializamos los pasos del agente en 0.
    steps = 0

    while True:
        # Obtenemos el estado actual.
        state = agent.get_state(game)
        # Generamos la acción correspondiente al estado actual.
        move = agent.get_action(state)
        # Ejecutamos la acción.
        reward, done, score = game.play_step(move)
        # TODO: Actividad 2.3: Actualiza la Q-Table.
        if done:
            # En caso de terminar el juego.

            # TODO: Actividad 2.4: Actualiza el Exploration Rate.

            # Reiniciamos el juego.
            game.reset()
            # Actualizamos los juegos jugados por el agente.
            agent.n_games += 1
            # Imprimimos el desempeño del agente cada 100 juegos.
            if agent.n_games % 100 == 0:
                # pickle.dump(agent, open("model/agent.p", "wb"))
                print('Game', agent.n_games, 'Mean Score', period_score /
                      100, 'Record:', record, "STEPS:", period_steps/100)
                # pickle.dump([plot_scores, plot_mean_scores], open("model/scores.p", "wb"))
                record = 0
                period_score = 0
                period_steps = 0

            # Actualizamos el record del agente.
            if score > record:
                record = score

            # Incrementamos nuestros indicadores.
            period_steps += steps
            period_score += score
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            steps = 0
            # plot(plot_scores, plot_mean_scores)
            if agent.n_games == NUM_EPISODES:
                break
        else:
            steps += 1


if __name__ == '__main__':
    train()
