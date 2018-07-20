import os
import yaml

class Constants:
  '''
  A meeting place for all things good and evil.
  '''

  @staticmethod
  def load_config(config_file):
    '''
    Loads the config file from disk and puts it into the Constants class
    '''
    with open(os.path.join('configs', config_file), 'r') as data_file:
      config = yaml.safe_load(data_file)
    config['learning_rate'] = float(config['learning_rate'])


    if config['dataset'] == 'cifar100':
      config['input_shape'] = [None, 32, 32, 3]
      config['num_total_classes'] = 100
    elif config['dataset'] == 'miniimagenet':
      config['input_shape'] = [None, 256, 256, 3]
      config['num_total_classes'] = 100
    else:
      raise ValueError('Dataset "{}" unknown'.format(config['dataset']))

    Constants.config = config

  @staticmethod
  def _cast_string_like(val, other):
    '''
    Casts the string `val` to the same type as `other`
    Returns None if the type of `other` isn't supported
    '''
    cast_val = None
    the_type = type(other)
    if the_type == int or the_type == float or the_type == bool or the_type == str:
      # Casting these types is the same
      cast_val = the_type(val)
    elif the_type == list:
      # Lists are a little special, but we'll only consider simple lists with non-mixed types
      # Split the line and remove the square-brackets on either side
      str_list = list(map(lambda x: x.strip(), val.split(',')))
      str_list[0] = str_list[0][1:]
      str_list[-1] = str_list[-1][:-1]
      # Get the type from the first element of the other
      inner_type = type(other[0])
      # Cast the contents of val to the right type
      cast_val = list(map(inner_type, str_list))

    return cast_val

  @staticmethod
  def _resolve_type_and_cast(key, value):
    '''
    Given a key and value, looks in the config dict to see if it exists.
    If so, it casts value to the same type as in the dict and returns it.
    If not, returns None
    '''
    cast_value = None
    if key in Constants.config:
      cast_value = Constants._cast_string_like(value, Constants.config[key])

    return cast_value

  @staticmethod
  def parse_args(argv):
    '''
    Reads command-line arguments and compares to the config dict. Replaces any existing keys and
    ignores all others
    '''
    if not argv:
      return
    # Positional arguments
    pos = []
    # Named arguments
    named = {}
    key = None
    for arg in argv:
      if key:
        if arg.startswith('--'):
          named[key] = True
          key = arg[2:]
        else:
          value = Constants._resolve_type_and_cast(key, arg)
          if value is not None:
            named[key] = value
          elif key in named:
            del named[key]
          key = None
      elif arg.startswith('--'):
        key = arg[2:]
      else:
        pos.append(arg)
    if key:
      named[key] = True
    Constants.config = {**Constants.config, **named}
