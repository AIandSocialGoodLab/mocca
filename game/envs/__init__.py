from gym.envs.registration import register

register(
    id='Pushing2D-v0',
    entry_point='envs.2Dpusher_env:Pusher2d'
)

register(
    id='Point2D-v0',
    entry_point='envs.2d_point:PointEnv'
)


register(
    id='Catcher2D-v0',
    entry_point='envs.2d_catcher:CatcherEnv'
)

register(
    id='Catcher2D-v1',
    entry_point='envs.2d_catcher_v1:CatcherEnv'
)

register(
	id='MultiAgent-Catcher2D-v0',
	entry_point='envs.2d_ma_catcher_v0:CatcherEnv'
)

register(
    id='MultiAgent-Catcher2D-v1',
    entry_point='envs.2d_ma_catcher_v1:CatcherEnv'
)

register(
    id='MultiAgent-Catcher2D-v2',
    entry_point='envs.2d_ma_catcher_v2:CatcherEnv'
)

register(
    id='MultiAgent-Catcher2D-v3',
    entry_point='envs.2d_ma_catcher_v3:CatcherEnv'
)

register(
    id='MultiAgent-Catcher2D-v5',
    entry_point='envs.2d_ma_catcher_v5:CatcherEnv'
)


register(
    id='MultiAgent-Catcher2D-v6',
    entry_point='envs.2d_ma_catcher_v6:CatcherEnv'
)

register(
    id='MultiAgent-Catcher2D-v7',
    entry_point='envs.2d_ma_catcher_v7:CatcherEnv'
)

register(
    id='MultiAgent-Catcher2DTest-v7',
    entry_point='envs.2d_ma_catcher_v7:CatcherEnv'
)

register(
    id='MultiAgent-Catcher2D-v8',
    entry_point='envs.2d_ma_catcher_v8:CatcherEnv'
)

register(
    id='MultiAgent-Catcher2DTest-v8',
    entry_point='envs.2d_ma_catcher_v8_test:CatcherEnv'
)





