import pygame
import pdb


class Movie(object):
    """A Movie playing class with pygame package
    """

    def __init__(self, filename):
        pygame.mixer.quit()
        self._movie = pygame.movie.Movie('./video/ex'+str(filename)+'.mpg')
        self.w, self.h = [size for size in self._movie.get_size()]
        self.mscreen = pygame.Surface((self.w, self.h)).convert()
        self._movie.set_display(self.mscreen, pygame.Rect(0, 0, int(self.w), int(self.h)))
        self._movie.play()

    def stop(self, delete=False):
        self._movie.stop()
        if delete:
            del self._movie
            pygame.mixer.init()

    def rewind(self):
        "replay the movie"
        self._movie.rewind()

    def position(self, screen_w, screen_h, scale=1):
        """ generate movie's position.
            According to the keyboard feedback may become larger or smaller.
        """
        # return [int(self.w*scale), int(self.h*scale)]
        return [screen_w-20-int(self.w*scale), screen_h-20-int(self.h*scale)]

    def draw(self, surface, screen_w, screen_h, scale=1, change=False):
        "Draw current frame to the surface"
        if change:
            self.mscreen = pygame.Surface((int(self.w*scale), int(self.h*scale))).convert()
            self._movie.set_display(self.mscreen, pygame.Rect(0, 0, int(self.w*scale), int(self.h*scale)))
        surface.blit(self.mscreen, self.position(screen_w, screen_h, scale))
