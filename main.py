from util import load_setting, write_setting, update_time

if __name__ == '__main__':
    data = load_setting('setting.json')
    data = update_time(data)
    write_setting(data,'')