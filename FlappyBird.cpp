#include <iostream>
#include <conio.h>   // for _kbhit() and _getch()
#include <windows.h> // for Sleep()
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

const int width = 40;
const int height = 20;
int birdY = height / 2;
int score = 0;
bool gameOver = false;

struct Pipe {
    int x;
    int gapY;
    int gapSize;
};

vector<Pipe> pipes;

void setup() {
    srand(time(0));
    pipes.push_back({width - 1, rand() % (height - 6) + 3, 5});
}

void draw() {
    system("cls");

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool printed = false;

            if (x == 0 || x == width - 1) {
                cout << "#";
                printed = true;
            } else if (x == 5 && y == birdY) {
                cout << "O";
                printed = true;
            } else {
                for (const auto& p : pipes) {
                    if (x == p.x && (y < p.gapY || y > p.gapY + p.gapSize)) {
                        cout << "|";
                        printed = true;
                        break;
                    }
                }
            }

            if (!printed) cout << " ";
        }
        cout << "\n";
    }
    cout << "Score: " << score << "\n";
}

void input() {
    if (_kbhit()) {
        char ch = _getch();
        if (ch == ' ' || ch == 'w') {
            birdY -= 3;
        }
    }
}

void logic() {
    birdY += 1;

    if (birdY <= 0 || birdY >= height - 1) {
        gameOver = true;
    }

    for (auto& p : pipes) {
        p.x--;

        if (p.x == 5) {
            if (birdY < p.gapY || birdY > p.gapY + p.gapSize) {
                gameOver = true;
            } else {
                score++;
            }
        }
    }

    if (pipes.size() > 0 && pipes[0].x < 0) {
        pipes.erase(pipes.begin());
    }

    if (pipes.empty() || pipes.back().x < width - 15) {
        pipes.push_back({width - 1, rand() % (height - 6) + 3, 5});
    }
}

int main() {
    setup();
    while (!gameOver) {
        draw();
        input();
        logic();
        Sleep(100);  // controls speed
    }
    cout << "\nGame Over! Final Score: " << score << "\n";
    return 0;
}
