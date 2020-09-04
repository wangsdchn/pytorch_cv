import trainer as Trainer
from utils.set_seeds import set_seeds

def main():
    trainer_name = 'Cifar10Trainer'
    trainer = Trainer.__dict__[trainer_name]()
    if trainer.args.phase == 'test':
        trainer.test(epoch=0)
        exit()
    for epoch in range(trainer.args.total_epochs):
        trainer.train(epoch)
        trainer.test(epoch)
        trainer.save_chackpoint(epoch)


if __name__ == '__main__':
    set_seeds(0)
    main()