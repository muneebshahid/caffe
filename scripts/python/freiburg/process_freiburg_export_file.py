import filesystem_helper as fs

summer = []
winter = []

source_folder = 'SOURCE_FOLDER'
target_folder = 'TARGET_FOLDER'

target_folder_winter = target_folder + '/dataset/winter/'
target_folder_summer = target_folder + '/dataset/summer/'
copy_files = True

if copy_files:
	if fs.path_exists(target_folder_winter):
		fs.rm_dir(target_folder_winter)
	if fs.path_exists(target_folder_summer):
		fs.rm_dir(target_folder_summer)

	fs.make_dir(target_folder_winter)
	fs.make_dir(target_folder_summer)



with open(source_folder + '/' + 'exported_season_match.txt', "r") as lines:
	for line_num, line in enumerate(lines):
		line_arr = line.split(' ')
		len_arr = len(line_arr)
		if len_arr > 3:
			im_file = line_arr[1]
			fs.copy(im_file, target_folder_summer + fs.get_file_name(im_file))		
			i = 2
			while i + 1 < len_arr:
				summer.append(line_arr[0] + ',' + line_arr[1])
				winter.append(line_arr[i] + ',' + line_arr[i + 1])
				im_file = line_arr[i + 1]
				fs.copy(im_file, target_folder_winter + fs.get_file_name(im_file))
				i = i + 2
		if line_num % 100 == 0:
			print str(line_num)

target_files = {'winter.txt': winter, 'summer.txt': summer}
for t_file in target_files:
	with open(target_folder + t_file, 'w') as f:
		for line in target_files[t_file]:
			f.write("%s\n" % line)