#!/bin/bash
for i in {1..20}
do
    python3 simulator.py --player_1 student_agent --player_2 random_agent --board_size 8
done