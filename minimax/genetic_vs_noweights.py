from genetic_algorithm import play_one_game

victoriesFor1 = 0
victoriesFor2 = 0

for i in range(50):
    playe1_weights = (2.969179069913862, 4.94362093706807, 3.2740160996771435)
    playe2_weights = (1, 1, 1)
    result = play_one_game(playe1_weights, playe2_weights, 4)
    if result == 1:
        victoriesFor1 += 1
    elif result == 2:
        victoriesFor2 += 1
    elif result == "draw":
        victoriesFor1 += 0.5
        victoriesFor2 += 0.5
    result = play_one_game(playe2_weights, playe1_weights, 4)
    if result == 1:
        victoriesFor1 += 1
    elif result == 2:
        victoriesFor2 += 1
    elif result == "draw":
        victoriesFor1 += 0.5
        victoriesFor2 += 0.5

print("Player 1 wins: ", victoriesFor1)
print("Player 2 wins: ", victoriesFor2)
