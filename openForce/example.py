import force_analyzer as fa

def main():
    force_cls = fa.forceAnalyzer('ForcePlate\\','test.csv')
    motion_cls = fa.motionAnalyzer('optitrack\\','test.c3d')
    # force_cls.get_action_point(scale=2.0)
    force_cls.plot()
    print(force_cls.get_peek_time())
    peek_time = force_cls.get_peek_time()
    motion_cls.set_peek_time(peek_time)
    motion_cls.plot()

if __name__ == '__main__':
    main()
