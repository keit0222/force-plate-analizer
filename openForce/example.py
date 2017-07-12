import force_analyzer as fa
def main():
    force_cls = fa.forceAnalyzer('ForcePlate\\','test.csv')
    motion_cls = fa.motionAnalyzer('optitrack\\','test.c3d')
    # force_cls.plot()
    peek_time = force_cls.get_peek_time()
    motion_cls.set_peek_time(peek_time)

    force_plate_action_points = force_cls.get_peek_action_point_for_converting()

    # force_cls.plot_peek_action_point()
    motion_action_points = motion_cls.get_action_point()

    action_points = []
    for force,motion in zip(force_plate_action_points,motion_action_points):
        action_points.append([force,motion])

    print(action_points)
    # motion_cls.plot()

if __name__ == '__main__':
    main()
