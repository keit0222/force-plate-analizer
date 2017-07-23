import calibrate
import synchronize

def main():
    calibrate.Calibrate('..\\..\\20170721\\forcePlate\\','calib01.csv','..\\..\\20170721\\motive\\c3d\\','calibrate00.c3d','hummer')
    synchronize.Synchro('..\\..\\20170721\\forcePlate\\')

if __name__ == '__main__':
    main()
