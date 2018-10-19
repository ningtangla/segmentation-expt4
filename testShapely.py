import shapely.geometry
import shapely.affinity
import pygame

rec = shapely.geometry.box(-1, -2.4, 1, 2.4)
rotateRec =shapely.affinity.rotate(rec, 30)
absRec = shapely.affinity.translate(rotateRec, 5, 8)
screen = pygame.display.set_mode([20, 20])
__import__('ipdb').set_trace()
screen.fill((255, 255, 255))
pygame.draw.polygon(screen, (255,0,0), absRec, 2)
