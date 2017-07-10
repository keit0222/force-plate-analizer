import force_analyzer as fa

def main():
    force_cls = fa.forceAnalyzer('ForcePlate\\','a03.csv')
    force_cls.get_action_point(scale=2.0)
    force_cls.plot()

if __name__ == '__main__':
    main()
