import os_helper as osh
summer = []
winter = []

proj_home = osh.get_env_var('PROJ_HOME')
source_folder = proj_home + '/data/test/freiburg/'
with open(source_folder + 'exported_season_match.txt', "r") as lines:
    processed_exported_season_match = []
    for line_num, line in enumerate(lines):
        line_arr = line.split(' ')
        len_arr = len(line_arr)
        if len_arr > 3:
            instance = line_arr[1]
            i = 2
            while i + 1 < len_arr:
                instance += " " + line_arr[i + 1]
                i += 2
            if line_num % 100 == 0:
                print str(line_num)
            processed_exported_season_match.append(instance)
    with open(source_folder + 'processed_season_match.txt', 'w') as f:
        f.writelines([line + '\n' for line in processed_exported_season_match])