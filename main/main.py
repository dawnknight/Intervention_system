from .klib import bodygame

# __main__ = "Kinect v2 Body Analysis"
def main():
    game = bodygame.BodyGameRuntime()
    game.run()