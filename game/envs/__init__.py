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





