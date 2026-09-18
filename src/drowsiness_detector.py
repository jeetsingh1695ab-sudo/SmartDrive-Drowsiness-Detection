# SmartDrive - Drowsiness Detection

DROWSY_TIME = 2.0

def check_drowsiness(closed_duration):
    if closed_duration >= DROWSY_TIME:
        return 'DROWSY'
    elif closed_duration > 0:
        return 'WARNING'
    else:
        return 'SAFE'


def update_eye_status(left_state, right_state):
    return left_state == 'CLOSED' and right_state == 'CLOSED'


if __name__ == '__main__':
    print('SmartDrive Drowsiness Detector')
    print('Drowsiness threshold:', DROWSY_TIME, 'seconds')