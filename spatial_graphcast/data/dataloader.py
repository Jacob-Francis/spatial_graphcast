

class MyNewDataLoader(data_loader_base.DataLoader):
  def __init__(
      self,
      *args,
      interpolation: Optional[interpolations.Interpolation] = None,
      compute: bool = True,
      add_nan_mask: bool = False,
  ):
    super().__init__(
        interpolation=interpolation,
        compute=compute,
        add_nan_mask=add_nan_mask,
    )

  def _load_chunk_from_source(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    if not isinstance(lead_times, np.ndarray):
      raise ValueError('Only exact lead times are supported.')

    datasets = []
    for init_time in init_times:
      for lead_time in lead_times:
        ds = some_data_loading_function(init_time, lead_time)
        datasets.append(ds)
    chunk = xr.merge(datasets)
    return chunk