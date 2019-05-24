import matplotlib


def display_small_cloud (cloud ):
    fig = matplotlib.pyplot.figure ()
    ax = matplotlib.Axes3D(fig)

    for i in range(0, cloud.size):
        ax.scatter(cloud[i][0], cloud[i][1], cloud[i][2])

    matplotlib.pyplot.show ()
