import os_helper as osh

summer = []
winter = []

proj_home = osh.get_env_var('PROJ_HOME')
exported_season_match = proj_home + '/data/test/unprocessed/freiburg/exported_season_match.txt'
source_folder = proj_home + '/data/test/unprocessed/freiburg/'
target_folder = proj_home + '/data/test/processed/freiburg/'

target_folder_winter = target_folder + 'dataset/winter/'
target_folder_summer = target_folder + 'dataset/summer/'
copy_files = True

if copy_files:
    if osh.path_exists(target_folder_winter):
        osh.rm_dir(target_folder_winter)
    if osh.path_exists(target_folder_summer):
        osh.rm_dir(target_folder_summer)
    osh.make_dir(target_folder_winter)
    osh.make_dir(target_folder_summer)

with open(exported_season_match, "r") as lines:
    processed_exported_season_match = []
    for line_num, line in enumerate(lines):
        line_arr = line.split(' ')
        len_arr = len(line_arr)
        if len_arr > 3:
            im_file = source_folder + line_arr[1]
            destination_summer = target_folder_summer + osh.extract_name_from_path(im_file)
            osh.copy(im_file, destination_summer)
            i = 2
            while i + 1 < len_arr:
                im_file = source_folder + line_arr[i + 1]
                destination_winter = target_folder_winter + osh.extract_name_from_path(im_file)
                osh.copy(im_file, destination_winter)
                processed_exported_season_match.append([destination_summer, destination_winter])
                i += 2
                if line_num % 100 == 0:
                    print str(line_num)
    with open(target_folder + 'processed_season_match.txt', 'w') as f:
        f.writelines([line[0] + ' ' + line[1] + '\n' for line in processed_exported_season_match])