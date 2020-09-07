install.packages("tidyverse") 
library(tidyverse)
#> ?????? Attaching packages ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????? tidyverse 1.3.0 ??????
#> ??? ggplot2 3.3.0     ??? purrr   0.3.4
#> ??? tibble  3.0.1     ??? dplyr   0.8.5
#> ??? tidyr   1.0.3     ??? stringr 1.4.0
#> ??? readr   1.3.1     ??? forcats 0.5.0
tidyverse_update() # Update if necessary

install.packages(c("nycflights13", "gapminder", "Lahman"))
# Data Packages
# airline flights, world development and baseball

#> ?????? Data Visualisation  ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
# Lets start with looking at a car data set which found in ggplot2 package found 
# in the tidyverse package.

mpg  # data frame
?mpg # information about the data frame

# There are 234 rows and 11 columns in this data set
# The variables are:
#> manufacturer: the manufacturer name
#> model: the models name
#> displ: the engine displacement, in litre
#> year: year of manufacture
#> cyl: number of cylinders
#> tran: type of transmission
#> drv: the type of driven train
#> cty: city miles per gallon
#> hwy: highway miles per gallon
#> fl: fuel type
#> class: 'type' of car

# Variable of interest from this data frame is 'displ' and 'hwy',

ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy))

# The plot illustrate a negative relationship between 'displ' and 'hwy' as the
# 'hwy' decrease when 'displ' increase.
# From this plot we can see that cars with big engines use more fuel.


# For data type which are objects it is also possible to visualize those
# together with the integer/float data - by color the plot

ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy, color = class))

# or sizes

ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy, size = class))
#> Warning: Using size for a discrete variable is not advised.

# or alpha, the transparency

ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy, alpha = class))

# or shapes

ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy, shape = class))

# ggplot2 can only visualize 6 different shapes. If there are more 
# classes/groups they will not be plotted


# Facets is another way to visualize categorical variables
# like 'class' in the mpg data set.

ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy)) + 
  facet_wrap(~ class, nrow = 2)

# facet_wrap() need a formula (~) which is name of a data 
# structure and the variable name

# If you want to plot two categorical variables you need to
# use the facet_grid()

ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy)) + 
  facet_grid(drv ~ cyl)


# geom have different syntax to show different visualization
# of the data

ggplot(data = mpg) + 
  geom_smooth(mapping = aes(x = displ, y = hwy))

# to remove the transparent field around the line use
# se = FALSE in the function

ggplot(data = mpg) + 
  geom_smooth(mapping = aes(x = displ, y = hwy, linetype = drv))

ggplot(data = mpg) +
  geom_smooth(mapping = aes(x = displ, y = hwy))

ggplot(data = mpg) +
  geom_smooth(mapping = aes(x = displ, y = hwy, group = drv))

ggplot(data = mpg) +
  geom_smooth(
    mapping = aes(x = displ, y = hwy, color = drv),
    show.legend = FALSE
  )


# YOu can display two plots in the same plot

ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy)) +
  geom_smooth(mapping = aes(x = displ, y = hwy))

# To make sure that the plots used the same x- and y-axis
# you can write it under the data

ggplot(data = mpg, mapping = aes(x = displ, y = hwy)) + 
  geom_point() + 
  geom_smooth()

# or like this

ggplot(data = mpg, mapping = aes(x = displ, y = hwy)) + 
  geom_point(mapping = aes(color = class)) + 
  geom_smooth()

