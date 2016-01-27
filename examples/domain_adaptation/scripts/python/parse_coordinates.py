import os_helper as osh


def read_file():
    with open(coord_file_txt, 'r') as coord_file_handle:
        lines = coord_file_handle.readlines()
    return lines


def split_qu_db(data):
    for pair in data:
        qu, db = pair.split(',')
        qu, db = qu.split(' '), db.replace('\n', '').split(' ')


def main():
    qu, db = split_qu_db(read_file())
    return

if __name__ == '__main__':
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    coord_file_txt = caffe_root + '/data/domain_adaptation_data/images/coordinates.txt'
    main()